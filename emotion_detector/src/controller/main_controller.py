from controller.pygame_controller import PygameController


class MainController:
    def __init__(self) -> None:
        pass

    def main_loop(self) -> None:
        pygame_controller = PygameController()
        pygame_controller.pygame_loop()
