from controller.pygame_controller import PygameController
from helpers.utils import blockPrint, enablePrint


class MainController:
    def __init__(self) -> None:
        pass

    def main_loop(self) -> None:
        blockPrint()
        pygame_controller = PygameController()
        pygame_controller.pygame_loop()
        enablePrint()
