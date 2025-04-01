window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  if (image) {
    image.ondragstart = function() { return false; };
    image.oncontextmenu = function() { return false; };
    $('#interpolation-image-wrapper').empty().append(image);
  }
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");
    });

    // Updated Carousel configuration
    var carouselOptions = {
        slidesToScroll: 3,
        slidesToShow: 3,
        loop: true,
        infinite: true,
        autoplay: false,
        autoplaySpeed: 3000,
        pagination: true,
        navigationSwipe: true,
        navigationKeys: true,
        breakpoints: [
            {
                changePoint: 1024,
                slidesToShow: 3,
                slidesToScroll: 3
            },
            {
                changePoint: 768,
                slidesToShow: 2,
                slidesToScroll: 2
            },
            {
                changePoint: 480,
                slidesToShow: 1,
                slidesToScroll: 1
            }
        ]
    };

    // Initialize carousels
    var carousels = bulmaCarousel.attach('.carousel', carouselOptions);

    // Handle video playback when slide changes
    carousels.forEach(carousel => {
        // Play all visible videos initially
        function playVisibleVideos() {
            const visibleSlides = carousel.element.querySelectorAll('.is-active');
            visibleSlides.forEach(slide => {
                const video = slide.querySelector('video');
                if (video) {
                    // Reset the video to start and play
                    video.currentTime = 0;
                    const playPromise = video.play();
                    if (playPromise !== undefined) {
                        playPromise.catch(error => {
                            console.log("Auto-play was prevented:", error);
                        });
                    }
                }
            });
        }

        // Play videos on initial load
        playVisibleVideos();

        // Handle slide changes
        carousel.on('after:show', state => {
            setTimeout(playVisibleVideos, 100); // Small delay to ensure DOM is updated
        });

        // Add click handlers for navigation buttons
        const prevButton = carousel.element.querySelector('.previous');
        const nextButton = carousel.element.querySelector('.next');
        
        if (prevButton) {
            prevButton.addEventListener('click', () => {
                setTimeout(playVisibleVideos, 100);
            });
        }
        
        if (nextButton) {
            nextButton.addEventListener('click', () => {
                setTimeout(playVisibleVideos, 100);
            });
        }
    });

    // Preload interpolation images if they exist
    if (typeof INTERP_BASE !== 'undefined') {
        preloadInterpolationImages();
        
        $('#interpolation-slider').on('input', function(event) {
            setInterpolationImage(this.value);
        });
        setInterpolationImage(0);
        $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);
    }

    // Initialize sliders
    if (typeof bulmaSlider !== 'undefined') {
        bulmaSlider.attach();
    }
});
