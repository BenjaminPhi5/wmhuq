class VPrint:
    def __init__(self, verbose=False):
        self.verbose = verbose
    def __call__(self, text):
        if self.verbose:
            print(text)
