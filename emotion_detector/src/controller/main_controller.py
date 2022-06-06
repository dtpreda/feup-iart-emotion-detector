from controller.pygame_controller import PygameController
from helpers.utils import block_print, enable_print


class MainController:
    def __init__(self) -> None:
        pass

    def main_loop(self) -> None:
        block_print()
        pygame_controller = PygameController()
        pygame_controller.pygame_loop()
        enable_print()
