import pygame

BG_COLOR = (252, 252, 220)
FONT_TITLE_SIZE = 50
FONT_SUBTITLE_SIZE = 35
FONT_TEXT_SIZE = 25
TEXT_COLOR = (20, 20, 20)


class MainView:
    def __init__(self, window, textarea, window_size) -> None:
        self.window = window
        self.window_size = window_size
        self.textarea = textarea

    def draw(self, events, emotion, train_accuracy, train_precision, train_recall,
             test_accuracy, test_precision, test_recall, duration):
        pressed_keys = pygame.key.get_pressed()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()

        self.window.fill(BG_COLOR)

        self.textarea.display_editor(
            events, pressed_keys, mouse_x, mouse_y, mouse_pressed)

        pygame.draw.line(self.window, (150, 150, 150),
                         (50, 500), (950, 500), 4)

        pygame.draw.line(self.window, (150, 150, 150),
                         (950, 200), (950, 750), 4)

        self.draw_infos(emotion, train_accuracy, train_precision, train_recall,
                        test_accuracy, test_precision, test_recall, duration)

    def draw_infos(self, emotion, train_accuracy, train_precision, train_recall,
                   test_accuracy, test_precision, test_recall, duration):
        font = pygame.font.SysFont(None, FONT_TEXT_SIZE)
        img = font.render('Duration -> ' +
                          str(duration) + (' seconds' if duration != '' and duration != 'analysing...' else ''), True, TEXT_COLOR)
        self.window.blit(
            img, (50, 730))

        font = pygame.font.SysFont(None, FONT_TEXT_SIZE)
        img = font.render('On 20% of the train dataset -> accuracy:' +
                          str(train_accuracy) + "; precision: " +
                          str(train_precision)
                          + "; recall: " + str(train_recall), True, TEXT_COLOR)
        self.window.blit(
            img, (50, 630))

        font = pygame.font.SysFont(None, FONT_TEXT_SIZE)
        img = font.render("On the test dataset -> accuracy: " +
                          str(test_accuracy) + "; precision: " +
                          str(test_precision)
                          + "; recall: " + str(test_recall), True, TEXT_COLOR)
        self.window.blit(img, (50, 680))

        font = pygame.font.SysFont(None, FONT_TEXT_SIZE)
        img = font.render("Emotion: " + emotion, True, TEXT_COLOR)
        self.window.blit(
            img, (550, 430))

        font = pygame.font.SysFont(None, FONT_TEXT_SIZE)
        img = font.render("Text Input", True, TEXT_COLOR)
        self.window.blit(
            img, (50, 200))

        font = pygame.font.SysFont(None, FONT_TITLE_SIZE)
        img = font.render("Natural Language Processing", True, TEXT_COLOR)
        self.window.blit(
            img, ((self.window_size[0] - img.get_size()[0]) / 2, 40))

        font = pygame.font.SysFont(None, FONT_SUBTITLE_SIZE)
        img = font.render(
            "Detection of emotions in small texts", True, TEXT_COLOR)
        self.window.blit(
            img, ((self.window_size[0] - img.get_size()[0]) / 2, 100))
