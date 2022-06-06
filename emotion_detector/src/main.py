from controller.pygame_controller import PygameController
from helpers.utils import block_print, enable_print

if __name__ == "__main__":
    block_print()
    pygame_controller = PygameController()
    pygame_controller.pygame_loop()
    enable_print()
