from __future__ import division
import sys
from timeit import default_timer
from tempfile import NamedTemporaryFile
from matplotlib import pyplot as plt
from IPython.display import Image, HTML, display, display_png


def anim_to_html(anim, fps=6, dpi=30, writer='ffmpeg', fmt='mp4'):
    """
    Convert a matplotlib animation object to a video embedded in an HTML
    <video> tag.

    Uses avconv (default) or ffmpeg.

    Returns an IPython.display.HTML object for embedding in the notebook.

    Adapted from `the yt project docs
    <http://yt-project.org/doc/cookbook/embedded_webm_animation.html>`__.
    """
    VIDEO_TAG = """
    <video controls>
    <source src="data:video/{fmt};base64,{data}" type="video/{fmt}">
    Your browser does not support the video tag.
    </video>"""
    plt.close(anim._fig)
    extra_args = []
    if writer == 'avconv':
        extra_args.extend(['-vcodec', 'libvpx'])
    if writer == 'ffmpeg':
        extra_args.extend(['-vcodec', 'libx264'])
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.' + fmt) as f:
            anim.save(f.name, fps=fps, dpi=dpi, writer=writer,
                      extra_args=extra_args)
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    return HTML(VIDEO_TAG.format(fmt=fmt, data=anim._encoded_video))


def format_time(t):
    """Format seconds into a human readable form.

    >>> format_time(10.4)
    '10.4s'
    >>> format_time(1000.4)
    '16min 40.4s'
    """
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    if h:
        return '{0:2.0f}hr {1:2.0f}min {2:4.1f}s'.format(h, m, s)
    elif m:
        return '{0:2.0f}min {1:4.1f}s'.format(m, s)
    else:
        return '{0:4.1f}s'.format(s)


def progressbar(it, stream=sys.stdout, size=50):
    count = len(it)
    start_time = default_timer()

    def _show(_i):
        tics = int(size*_i/count)
        bar = '#' * tics
        percent = (100 * _i) // count
        elapsed = format_time(default_timer() - start_time)
        msg = '\r[{0:<{1}}] | {2}% Completed | {3}'.format(bar, size,
                                                           percent, elapsed)
        stream.write(msg)
        stream.flush()

    _show(0)
    for i, item in enumerate(it):
        yield item
        _show(i+1)
    stream.write("\n")
    stream.flush()


