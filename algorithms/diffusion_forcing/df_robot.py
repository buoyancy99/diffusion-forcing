from omegaconf import DictConfig
import torch
import numpy as np
import pyrealsense2 as rs
import zmq
import cv2
from einops import rearrange
from torchvision.transforms import Resize

from algorithms.diffusion_forcing.df_video import DiffusionForcingVideo
from utils.logging_utils import log_video, get_validation_metrics_for_videos
from utils.robot_utils import unpack_to_1d


class DiffusionForcingRobot(DiffusionForcingVideo):
    """
    Robot imitation learning with Diffusion Forcing
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cameras = []
        self.socket = None
        self.n_cameras = cfg.n_cameras

    def training_step(self, batch, batch_idx):
        # if batch_idx == 0:
        #     self.visualize_noise(batch)

        output_dict = super().training_step(batch, batch_idx)
        xs_pred = output_dict["xs_pred"]
        xs = output_dict["xs"]
        video_pred = torch.cat([xs_pred[:, :, :3], xs_pred[:, :, 3:6]], -2)
        video = torch.cat([xs[:, :, :3], xs[:, :, 3:6]], -2)

        if batch_idx % 5000 == 0:
            log_video(
                video_pred,
                video,
                step=self.global_step,
                namespace="training_vis",
                logger=self.logger.experiment,
            )
        return output_dict

    def on_validation_epoch_end(self, namespace="validation"):
        if not self.validation_step_outputs:
            return

        xs_pred = []
        xs = []
        for pred, gt in self.validation_step_outputs:
            xs_pred.append(pred)
            xs.append(gt)
        xs_pred = torch.cat(xs_pred, 1)
        xs = torch.cat(xs, 1)

        video_pred = torch.cat([xs_pred[:, :, :3], xs_pred[:, :, 3:6]], -2)
        video = torch.cat([xs[:, :, :3], xs[:, :, 3:6]], -2)

        log_video(
            video_pred,
            video,
            step=None if namespace == "test" else self.global_step,
            namespace=namespace + "_vis",
            context_frames=self.context_frames,
            logger=self.logger.experiment,
        )

        metric_dict = get_validation_metrics_for_videos(
            video_pred[self.context_frames :],
            video[self.context_frames :],
            lpips_model=self.validation_lpips_model,
            fid_model=self.validation_fid_model,
            fvd_model=self.validation_fvd_model,
        )
        self.log_dict(
            {f"{namespace}/{k}": v for k, v in metric_dict.items()}, on_step=False, on_epoch=True, prog_bar=True
        )

        self.validation_step_outputs.clear()

    def maybe_reset_cameras(self):
        if not self.cameras:
            ctx = rs.context()
            devices = ctx.query_devices()
            for device in devices:
                serial = device.get_info(rs.camera_info.serial_number)
                camera = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                camera.start(config)
                self.cameras.append(camera)
        if len(self.cameras) != self.n_cameras:
            raise RuntimeError(f"Expected {self.n_cameras} cameras, but found {len(self.cameras)}")

    def maybe_reset_socket(self):
        if not self.socket:
            ctx = zmq.Context()
            self.socket = ctx.socket(zmq.REP)
            self.socket.bind(self.cfg.robot.address)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if self.frame_stack > 1:
            raise NotImplementedError("frame_stack > 1 not implemented for robot dataset")
        dummy_x = batch[0]
        max_steps = dummy_x.shape[1]
        self.maybe_reset_cameras()
        self.maybe_reset_socket()
        n_cameras = len(self.cameras)
        print(f"Detected {n_cameras} cameras")

        resize = Resize(self.x_shape[-2:], antialias=True)

        while True:
            z = torch.zeros(1, *self.z_shape)
            z = z.to(self.device)
            action_stack = len(self.data_mean) - n_cameras * 3
            a = dummy_x[0][0][-action_stack:]  # take action padding from data
            a = a[None]

            for _ in range(max_steps):
                # wait for robot request
                message = self.socket.recv()
                if message == b"stop":
                    self.socket.send(b"restarting")
                    print("Received stop message. Restarting...")
                    break
                else:
                    print(f"Received request: {message.decode()}")

                # read cameras
                o = []
                for cam in self.cameras:
                    frame = cam.wait_for_frames()
                    o.append(np.array(frame.get_color_frame().get_data()))
                if self.debug:
                    cv2.imwrite("debug.png", o[0])
                    input("Checkout debug.png to verify camera order is same as training data. ")

                o = torch.from_numpy(np.stack(o) / 255.0).float().permute(0, 3, 1, 2)
                o = resize(o).to(self.device)
                o = rearrange(o, "n c h w -> 1 (n c) h w")  # (n_cam * 3, h, w)
                x = torch.cat([o, a], 1)
                x = self._normalize_x(x)

                # update posterior
                z, _, _, _ = self.transition_model(z, x, None, deterministic_t=0)

                # predict next step
                _, x_pred = self.transition_model.rollout(z, None)
                x_pred = self._unnormalize_x(x_pred)
                o_pred, a = torch.split(x_pred, [3 * n_cameras, action_stack], 1)

                # send action
                actions = a[0].cpu().numpy()
                actions = [unpack_to_1d(sub) for sub in actions]
                actions = np.stack([np.concatenate([pos, quat, [grasp]]) for pos, quat, grasp in actions])
                message = actions.astype(np.float32).tobytes()
                self.socket.send(message)
