# ZeroGPU, the free Gradio hosting tier, refuses to start a Space that has no
# @spaces.GPU function. This app is CPU-only, so the placeholder below is never
# called and uses no GPU time. The `spaces` package exists only on Hugging Face,
# hence the guard.
try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_placeholder():
        return None
except ImportError:
    pass

from stitcher.web_app import demo

if __name__ == "__main__":
    demo.launch()