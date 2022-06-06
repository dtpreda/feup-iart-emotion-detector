from controller.pygame_controller import PygameController
from predictions.datasets_processing import process_dataset
from helpers.utils import blockPrint, enablePrint


class MainController:
    def __init__(self) -> None:
        pass

    def main_loop(self) -> None:
        # blockPrint()
        process_dataset()
        pygame_controller = PygameController()
        pygame_controller.pygame_loop()
        # enablePrint()
