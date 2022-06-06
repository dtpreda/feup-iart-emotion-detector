from threading import Thread
import pygame
import pygame_widgets
from pygame_texteditor import TextEditor
from time import time, sleep
from pygame_widgets.button import Button
from pygame_widgets.dropdown import Dropdown
from pygame_widgets.toggle import Toggle
from predictions.text_prediction import predict_text_emotion
from predictions.datasets_prediction import predict_dataset

from classifiers.generative import multi_layer_perceptron_fit, random_forest_fit, random_forest_predict, multi_layer_perceptron_predict
from classifiers.discriminative import gaussian_naive_bayes_fit, gaussian_naive_bayes_predict, multinomial_naive_bayes_fit, multinomial_naive_bayes_predict

from view.main_view import MainView
from helpers.utils import removeEmptyLines

WINDOW_SIZE = (1300, 800)
FPS = 30
BG_COLOR = (252, 252, 220)

# Text Area Constants
STYLE_FILE = "emotion_detector/asset/style/editor.yml"
FONT_SIZE = 25
OFFSET_X = 50
OFFSET_Y = 230
TEXTAREA_WIDTH = 850
TEXTAREA_HEIGHT = 150

# Buttons
COLOR_BUTTON = (190, 190, 190)
HOVER_BUTTON = (22, 174, 245)
HOVER_DROPDOWN = (150, 150, 150)


class PygameController:
    def __init__(self) -> None:
        self.window_size = WINDOW_SIZE
        self.window = self.init_pygame()
        self.textarea = self.init_textarea()
        self.main_view = MainView(self.window, self.textarea, self.window_size)
        self.submit_button = self.init_evaluate_button()
        self.dataset_button = self.init_dataset_button()
        self.dropdowns = self.init_dropdowns()
        self.toggles = self.init_toggles()
        self.animation = self.init_animation()
        self.animation.hide()
        self.emotion = ""
        self.train_accuracy = ""
        self.test_accuracy = ""
        self.duration = ""

    def pygame_loop(self) -> None:
        while True:
            start = time()
            events = pygame.event.get()

            if not self.is_running(events):
                break

            self.main_view.draw(events, self.emotion,
                                self.train_accuracy, self.test_accuracy, self.duration)
            pygame_widgets.update(events)
            pygame.display.flip()

            sleep(max(1.0/FPS - (time() - start), 0))
            continue

        self.quit_pygame()

    def force_flip(self):
        events = pygame.event.get()
        self.main_view.draw(events, self.emotion,
                            self.train_accuracy, self.test_accuracy, self.duration)
        pygame_widgets.update(events)
        pygame.display.flip()

    def evaluate_text(self):
        if self.dropdowns[0].getSelected() == None:
            self.animate_warning("Please select the desired algorithm")
            return

        if self.dropdowns[1].getSelected() == None:
            self.animate_warning("Please select the desired dataset")
            return

        input = removeEmptyLines(self.textarea.get_text_as_string())

        self.emotion = 'analysing...'
        self.force_flip()

        algorithms = self.get_algorithm(self.dropdowns[0].getSelected())

        self.emotion = predict_text_emotion(
            input, algorithms[0], algorithms[1],
            self.get_dataset_dir(self.dropdowns[1].getSelected()),
            self.toggles[0][1].getValue(),
            self.toggles[1][1].getValue(),
            self.toggles[2][1].getValue(),
            self.toggles[3][1].getValue(),
            self.toggles[4][1].getValue(),
            self.toggles[5][1].getValue())

    def evaluate_dataset(self):
        if self.dropdowns[0].getSelected() == None:
            self.animate_warning("Please select the desired algorithm")
            return

        if self.dropdowns[1].getSelected() == None:
            self.animate_warning("Please select the desired dataset")
            return

        self.train_accuracy = self.test_accuracy = 'analysing...'
        self.duration = 'analysing...'
        self.force_flip()

        algorithms = self.get_algorithm(self.dropdowns[0].getSelected())

        self.train_accuracy, self.test_accuracy, self.duration = predict_dataset(
            algorithms[0], algorithms[1],
            self.get_dataset_dir(self.dropdowns[1].getSelected()),
            self.toggles[0][1].getValue(),
            self.toggles[1][1].getValue(),
            self.toggles[2][1].getValue(),
            self.toggles[3][1].getValue(),
            self.toggles[4][1].getValue(),
            self.toggles[5][1].getValue())

    def is_running(self, events) -> bool:
        """
        Check if does not exist at least one of the following events: press ESQ or close the window

        Return:
            False if the events exist, True otherwise 
        """
        for event in events:
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
        return True

    def init_pygame(self) -> None:
        """
        Pygame and screen initialization 

        Return:
            Pygame Surface to display
        """
        pygame.init()
        return pygame.display.set_mode(self.window_size)

    def quit_pygame(self) -> None:
        '''Unitialize all pygame modules'''
        pygame.quit()

    def init_textarea(self):
        textarea = TextEditor(
            OFFSET_X, OFFSET_Y, TEXTAREA_WIDTH, TEXTAREA_HEIGHT, self.window)
        textarea.set_font_size(FONT_SIZE)
        textarea.set_syntax_highlighting(True)
        textarea.set_colorscheme_from_yaml(STYLE_FILE)
        return textarea

    def init_evaluate_button(self):
        return Button(
            self.window, 150, 415, 150, 50,
            text='Evaluate',
            fontSize=30,
            margin=20,
            inactiveColour=COLOR_BUTTON,
            hoverColour=HOVER_BUTTON,
            pressedColour=HOVER_BUTTON,
            radius=20,
            onClick=self.evaluate_text
        )

    def init_dataset_button(self):
        return Button(
            self.window, 50, 550, 200, 50,
            text='Evaluate Dataset',
            fontSize=25,
            margin=20,
            inactiveColour=COLOR_BUTTON,
            hoverColour=HOVER_BUTTON,
            pressedColour=HOVER_BUTTON,
            radius=20,
            onClick=self.evaluate_dataset
        )

    def init_dropdowns(self):
        return (Dropdown(
            self.window, 1000, 200, 250, 50, name='Algorithm   v',
            choices=[
                'Gaussian Naive Bayes',
                'Multinomial Naive Bayes',
                'Random Forest',
                'Multilayer Perceptron'
            ],
            borderRadius=3,
            colour=pygame.Color(190, 190, 190),
            inactiveColour=COLOR_BUTTON,
            hoverColour=HOVER_DROPDOWN,
            pressedColour=HOVER_DROPDOWN,
            values=[0, 1, 2, 3],
            direction='down',
            textHAlign='centre',
            fontSize=25,
        ), Dropdown(
            self.window, 1000, 275, 250, 50, name='Train Dataset   v',
            choices=[
                'Dataset 1',
                'Dataset 2',
            ],
            borderRadius=3,
            colour=pygame.Color(190, 190, 190),
            inactiveColour=COLOR_BUTTON,
            hoverColour=HOVER_DROPDOWN,
            pressedColour=HOVER_DROPDOWN,
            values=[0, 1],
            direction='down',
            textHAlign='centre',
            fontSize=25,
        ))

    def init_toggles(self):
        return [
            (self.init_toggle_button("Remove Stop Words", 1000, 350),
             Toggle(self.window, 1225, 367, 30, 15)),
            (self.init_toggle_button("Lowercase", 1000, 425),
             Toggle(self.window, 1225, 442, 30, 15)),
            (self.init_toggle_button("Lemmatize", 1000, 500),
             Toggle(self.window, 1225, 517, 30, 15)),
            (self.init_toggle_button("Remove Single Chars", 1000, 575),
             Toggle(self.window, 1225, 592, 30, 15)),
            (self.init_toggle_button("Bigram", 1000, 650),
             Toggle(self.window, 1225, 667, 30, 15)),
            (self.init_toggle_button("PosTag", 1000, 725),
             Toggle(self.window, 1225, 742, 30, 15))
        ]

    def init_toggle_button(self, button_text, x, y):
        return Button(
            self.window, x, y, 200, 50,
            text=button_text,
            fontSize=25,
            margin=20,
            inactiveColour=BG_COLOR,
            hoverColour=BG_COLOR,
            pressedColour=BG_COLOR,
            radius=20,
        )

    def init_animation(self):
        return Button(
            self.window, 300, 700, 200, 50,
            text="Warning",
            fontSize=25,
            margin=20,
            inactiveColour=(200, 0, 0),
            hoverColour=(200, 0, 0),
            pressedColour=(200, 0, 0),
            radius=20,
            textColour=(255, 255, 255)
        )

    def animate_warning(self, warning_text):
        self.animation = Button(
            self.window, 450, 150, 400, 50,
            text=warning_text,
            fontSize=25,
            margin=20,
            inactiveColour=(220, 0, 0),
            hoverColour=(200, 0, 0),
            pressedColour=(200, 0, 0),
            radius=20,
            textColour=(255, 255, 255)
        )

        t = Thread(target=self.anim, args=())
        t.start()

    def anim(self):
        self.animation.show()
        sleep(2)
        self.animation.hide()

    def get_algorithm(self, value):
        if (value == 0):
            return (gaussian_naive_bayes_fit, gaussian_naive_bayes_predict)
        if (value == 1):
            return (multinomial_naive_bayes_fit, multinomial_naive_bayes_predict)
        if (value == 2):
            return (random_forest_fit, random_forest_predict)
        if (value == 3):
            return (multi_layer_perceptron_fit, multi_layer_perceptron_predict)

    def get_dataset_dir(self, value):
        if (value == 0):
            return "emotion_detector/dataset/twitter/"
        if (value == 1):
            return "emotion_detector/dataset/twitter2/"
