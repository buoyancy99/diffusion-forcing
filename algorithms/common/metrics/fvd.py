"""
Adopted from https://github.com/cvpr2022-stylegan-v/stylegan-v
Verified to be the same as tf version by https://github.com/universome/fvd-comparison
"""

import io
import re
import requests
import html
import hashlib
import urllib
import urllib.request
from typing import Any, List, Tuple, Union, Dict
import scipy

import torch
import torch.nn as nn
import numpy as np


def open_url(
    url: str,
    num_attempts: int = 10,
    verbose: bool = True,
    return_filename: bool = False,
) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match("^[a-z]+://", url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith("file://"):
        filename = urllib.parse.urlparse(url).path
        if re.match(r"^/[a-zA-Z]:", filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [
                                html.unescape(link)
                                for link in content_str.split('"')
                                if "export=download" in link
                            ]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError(
                                "Google Drive download quota exceeded -- please try again later"
                            )

                    match = re.search(
                        r'filename="([^"]*)"',
                        res.headers.get("Content-Disposition", ""),
                    )
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)


def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(
        np.dot(sigma_gen, sigma_real), disp=False
    )  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0)  # [d]
    sigma = np.cov(feats, rowvar=False)  # [d, d]

    return mu, sigma


class FrechetVideoDistance(nn.Module):
    def __init__(self):
        super().__init__()
        detector_url = (
            "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
        )
        # Return raw features before the softmax layer.
        self.detector_kwargs = dict(rescale=False, resize=True, return_features=True)
        with open_url(detector_url, verbose=False) as f:
            self.detector = torch.jit.load(f).eval()

    @torch.no_grad()
    def compute(self, videos_fake: torch.Tensor, videos_real: torch.Tensor):
        """
        :param videos_fake: predicted video tensor of shape (frame, batch, channel, height, width)
        :param videos_real: ground-truth observation tensor of shape (frame, batch, channel, height, width)
        :return:
        """
        n_frames, batch_size, c, h, w = videos_fake.shape
        if n_frames < 2:
            raise ValueError("Video must have more than 1 frame for FVD")

        videos_fake = videos_fake.permute(1, 2, 0, 3, 4).contiguous()
        videos_real = videos_real.permute(1, 2, 0, 3, 4).contiguous()

        # detector takes in tensors of shape [batch_size, c, video_len, h, w] with range -1 to 1
        feats_fake = self.detector(videos_fake, **self.detector_kwargs).cpu().numpy()
        feats_real = self.detector(videos_real, **self.detector_kwargs).cpu().numpy()

        return compute_fvd(feats_fake, feats_real)
