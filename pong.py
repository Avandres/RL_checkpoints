import pygame
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.optimizers import SGD
from copy import deepcopy
from skimage.transform import resize


def loss_function(q1, q2):
    return tf.reduce_sum(tf.square(q2 - q1))


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

FPS = 30


def screenshot(obj, size):
    img = pygame.Surface(size)
    img.blit(obj, (0, 0), ((0, 0), size))
    return np.array(pygame.PixelArray(img)).astype(bool).astype(int)


class Buttons():
    def __init__(self):
        self.KEYUP = False
        self.KEYDOWN = False
        self.UP = False
        self.DOWN = False


class Simple_AI_control():
	'''

	Простой алгоритм поведения без машинного обучения. Планка будет двигаться вниз, если 
	мяч находится ниже неё, и будет двигаться вверх, если мяч находится выше неё.
	
	'''
    def __init__(self, ball=None, plank=None):
        self.ball = ball
        self.plank = plank

    def get_command(self):
        command = 0
        if self.ball.rect.y < self.plank.rect.centery:
            command = 2
        elif self.ball.rect.y > self.plank.rect.centery:
            command = 1
        return command


class Deep_RL_control():
	'''

	Алгоритм через обучение с подкреплением. Планка двигается в ту сторону, в какую ей
	укажет двигаться нейронная сеть.

	'''
    def __init__(self, width, height, lr=0.00001, gamma=0.99, max_experience_len=200, train_part=0.3):
        self.gamma = gamma
        # For experience replay
        self.X_data = []
        self.y_data = []
        self.a_r = []
        self.MAX_EXPERIENCE_LEN = max_experience_len
        self.TRAIN_PART = train_part

        # model
        self.model = Sequential()
        self.model.add(Conv2D(input_shape=(width, height, 1),
                              filters=20,
                              kernel_size=(5, 5)))
        self.model.add(Flatten())
        self.model.add(Dense(200, activation="relu"))
        self.model.add(Dense(2, activation="sigmoid"))
        self.model.compile(optimizer=SGD(lr), loss=loss_function)

    def get_command(self, delta_frame):
    	'''

		Получение команды от нейронной сети (сеть обучена и применяется).

    	'''
        Q_predicted = self.model.predict(np.array(delta_frame)[None, ..., None])
        command = np.argmax(Q_predicted[0])
        return command+1

    def get_command_for_fit(self, delta_frame, r, is_done, action, is_checkpoint_loaded=False,
                            curr_ind=0):
    	'''

		Получение команды от нейронной сети в процессе обучения.

    	'''
        if is_checkpoint_loaded:
            self.X_data[curr_ind] = np.array(delta_frame)[None, ..., None]
            self.a_r[curr_ind] = (action, r)
        else:
            self.X_data.insert(0, np.array(delta_frame)[None, ..., None])
            self.a_r.insert(0, (action, r))

        if len(self.X_data) > self.MAX_EXPERIENCE_LEN:
            del self.X_data[-1]
            del self.a_r[-1]
        if is_done:
            for i, a_r in enumerate(self.a_r):
                a, r = a_r
                Q_predicted = self.model.predict(np.array(self.X_data[i]))
                Q_target = deepcopy(Q_predicted)
                if i == 0:
                    Q_target[0][a] = r
                else:
                    Q_target[0][action] = r + self.gamma * self.y_data[-1][0][action]
                self.y_data.append(Q_target)



    def fit(self, is_checkpoint_loaded=False, saves_len=0):
    	'''

    	Обучение нейронной сети.

    	'''
        X_data_for_random_choice = deepcopy(self.X_data)
        y_data_for_random_choice = deepcopy(self.y_data)
        X_train = []
        y_train = []
        X_train.append(X_data_for_random_choice[0][:, :])
        y_train.append(y_data_for_random_choice[0][:])
        del X_data_for_random_choice[0]
        del y_data_for_random_choice[0]
        i = 0
        while i < int(self.TRAIN_PART * len(X_data_for_random_choice)):
            ind = random.randint(0, len(X_data_for_random_choice)-1)
            X_train.append(X_data_for_random_choice[ind][:, :])
            y_train.append(y_data_for_random_choice[ind][:])
            del X_data_for_random_choice[ind]
            del y_data_for_random_choice[ind]
            i += 1
        if saves_len == 0:
            self.X_data = []
            self.a_r = []
        self.y_data = []
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        for j in range(len(y_train)):
            self.model.train_on_batch(X_train[j], y_train[j])



class Human_control():
	'''

	Класс, позволяющий пользователю контролировать движения одной из планок.

	'''
    def __init__(self, buttons_control):
        self.buttons_control = buttons_control
        self.command = 0

    def get_command(self):
        if self.buttons_control.KEYUP:
            self.command = 0
        if self.buttons_control.KEYDOWN:
            if self.buttons_control.UP:
                self.command = 2
            elif self.buttons_control.DOWN:
                self.command = 1
        return self.command


class Player_plank(pygame.sprite.Sprite):
	'''

	Класс для планок. Отвечает за сам объект (его существование и движение при получении команды).
	Алгоритмы поведения планок реализованы в других классах.

	'''
    def __init__(self, height, width, side=1, speed=1):
        self.speed = speed
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((width//55, height//10))
        self.image.fill(BLACK)
        self.rect = self.image.get_rect()
        if side == 1:
            self.rect.center = (width // 10, height // 2)
        if side == 2:
            self.rect.center = (width - width//10, height//2)

    def update(self, height, width, command):
        if command == 1:
            self.rect.y += self.speed * (height // 55)
        elif command == 2:
            self.rect.y -= self.speed * (height // 55)

        if self.rect.y > height-self.rect.height:
            self.rect.y = height-self.rect.height
        if self.rect.y < 0:
            self.rect.y = 0


class Ball(pygame.sprite.Sprite):
	'''

	Класс, реализующий поведение мяча на игровом поле (движение, столкновения).

	'''
    def __init__(self, height, width):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((width//60, height//60))
        self.image.fill(BLACK)
        self.rect = self.image.get_rect()
        self.rect.center = (width//2, height//2)

    def update(self, height, width, x_speed, y_speed):
        self.rect.x += x_speed * width // 100
        self.rect.y += y_speed * height // 100

        is_collide_with_wall = False
        if self.rect.y > height - self.rect.height:
            self.rect.y = height - self.rect.height
            is_collide_with_wall = True
        if self.rect.y < 0:
            self.rect.y = 0
            is_collide_with_wall = True

        if self.rect.x > width - self.rect.width:
            self.rect.x = width // 2
            self.rect.y = height // 2
        if self.rect.x < 0:
            self.rect.x = width // 2
            self.rect.y = height // 2
        return is_collide_with_wall


class Pong():
	'''

	Класс, реализующий игровую среду. Задаёт игровое поле, правила и т.п.

	'''
    def __init__(self, ball_speed,
                 player_1_speed,
                 player_2_speed,
                 width_of_window,
                 height_of_window,
                 with_checkpoints=False,
                 checkpoints_bias=30):
        self.ball_speed = ball_speed
        self.ball_speed_x = ball_speed
        self.ball_speed_y = 0

        self.shape_of_resize_image = (80, 80)

        self.width_of_window = width_of_window
        self.height_of_window = height_of_window
        self.width_of_plank = self.width_of_window // 10
        self.height_of_plank = self.height_of_window // 10

        self.width_of_ball = self.width_of_window // 10
        self.height_of_ball = self.height_of_window // 10

        self.player_1_speed = player_1_speed
        self.player_2_speed = player_2_speed

        self.player_1_y_coord = self.height_of_window // 2
        self.player_2_y_coord = self.height_of_window // 2

        self.ball_x_coord = self.width_of_window // 2
        self.ball_y_coord = self.height_of_window // 2

        self.player_1_score = 0
        self.player_2_score = 0

        self.frames_limit = checkpoints_bias
        self.with_saves = with_checkpoints
        self.LIMIT_FOR_SAVES = 50
        self.saves = []
        self.checkpoints_bias = checkpoints_bias
        self.is_checkpoint_loaded = False

    def save(self, ball, player_1, player_2):
    	'''

		Метод, позволяющий сохранять текущее состояние игры с последующей загрузкой.

    	'''
        save_dict = {
            "ball_speed": self.ball_speed,
            "ball_speed_x": self.ball_speed_x,
            "ball_speed_y": self.ball_speed_y,
            "ball_x_coord": ball.rect.centerx,
            "ball_y_coord": ball.rect.centery,
            "player_1_y_coord": player_1.rect.centery,
            "player_2_y_coord": player_2.rect.centery
        }
        self.saves.insert(0, save_dict)
        if len(self.saves) > self.LIMIT_FOR_SAVES:
            del self.saves[-1]

    def collide_upldate(self, player_group, ball, command):
    	'''

		Проверка всех столкновений, апдейт среды.

    	'''
        is_collide_with_player = pygame.sprite.spritecollide(ball, player_group, False)
        if is_collide_with_player:
            ball.rect.x = player_group.sprites()[-1].rect.x - player_group.sprites()[-1].rect.width * 1.1
            self.ball_speed *= -1

            if (command == 0):
                self.ball_speed_x *= -1

            if (command == 2) and (self.ball_speed_y < 0):
                self.ball_speed_x = self.ball_speed * 1 / 5
                self.ball_speed_y = self.ball_speed * 4 / 5
            elif (command == 2) and (self.ball_speed_y == 0):
                self.ball_speed_x = self.ball_speed * 1 / 2
                self.ball_speed_y = self.ball_speed * 1 / 2
            elif (command == 2) and (0 < self.ball_speed_y):
                self.ball_speed_x = self.ball_speed
                self.ball_speed_y = 0

            elif (command == 1) and (self.ball_speed_y > 0):
                self.ball_speed_x = self.ball_speed * 1 / 5
                self.ball_speed_y = -self.ball_speed * 4 / 5
            elif (command == 1) and (self.ball_speed_y == 0):
                self.ball_speed_x = self.ball_speed * 1 / 2
                self.ball_speed_y = -self.ball_speed * 1 / 2
            elif (command == 1) and (0 > self.ball_speed_y):
                self.ball_speed_x = self.ball_speed
                self.ball_speed_y = 0
            if ball.rect.x < self.width_of_window // 2:
                ball.rect.centerx = player_group.sprites()[-1].rect.centerx + player_group.sprites()[-1].rect.width * 2
            elif ball.rect.x >= self.width_of_window // 2:
                ball.rect.centerx = player_group.sprites()[-1].rect.centerx - player_group.sprites()[-1].rect.width * 2

    def reset(self, ball, player_1, player_2):
    	'''

		Сброс состояния среды. Планки и мяч оказываются на стартовых позициях. Используется,
		когда мяч залетел в чьи-то ворота, и начинается новый раунд.

    	'''
        self.ball_speed_x = self.ball_speed
        self.ball_speed_y = 0
        ball.rect.x = self.width_of_window // 2
        ball.rect.y = self.height_of_window // 2
        player_1.rect.centery = self.height_of_window // 2
        player_2.rect.centery = self.height_of_window // 2

    def load_checkpoint(self, ball, player_1, player_2):
    	'''
		
		Загрузка ранее сохранённого состояния среды. Нужно, чтобы при обучении не приходилось каждый
		раз начинать раунд заново.

    	'''
        self.is_checkpoint_loaded = True

        self.ball_speed = self.saves[0]["ball_speed"]
        self.ball_speed_x = self.saves[0]["ball_speed_x"]
        self.ball_speed_y = self.saves[0]["ball_speed_y"]
        ball.rect.x = self.saves[0]["ball_x_coord"]
        ball.rect.y = self.saves[0]["ball_y_coord"]
        player_1.rect.centery = self.saves[0]["player_1_y_coord"]
        player_2.rect.centery = self.saves[0]["player_2_y_coord"]

        self.frames_limit += 1
        del self.saves[0]

    def ball_final_coord(self, ball, player_1, player_2):
    	'''

		Проверка на то, залетел ли мяч за одну из планок (заработал ли один из игроков очко в текущем раунде)

    	'''
        is_done = False
        r = 0
        self.ball_y_coord = ball.rect.y
        self.ball_x_coord = ball.rect.x
        if self.ball_x_coord < player_1.rect.x - player_1.rect.width * 2:
            self.is_checkpoint_loaded = False
            self.reset(ball, player_1, player_2)
            self.player_2_score += 1
            print("player 1:", self.player_1_score, "| player 2:", self.player_2_score)
            r = 1
            is_done = True
        if self.ball_x_coord > player_2.rect.x + player_2.rect.width * 2:
            if not self.is_checkpoint_loaded:
                self.saves = [elem for i, elem in enumerate(self.saves) if i > self.checkpoints_bias]
                self.frames_limit = self.checkpoints_bias
                r = -1
            else:
                r = -0.01
            if len(self.saves) == 0:
                self.is_checkpoint_loaded = False
                self.frames_after_checkpoint = self.frames_limit
            self.player_1_score += 1
            print("player 1:", self.player_1_score, "| player 2:", self.player_2_score)
            is_done = True
        return r, is_done

    def main(self):		# Сделать метод компактнее!
        player_1 = Player_plank(self.height_of_window, self.width_of_window, side=1, speed=self.player_1_speed)
        player_2 = Player_plank(self.height_of_window, self.width_of_window, side=2, speed=self.player_2_speed)
        ball = Ball(self.height_of_window, self.width_of_window)
        buttons_control = Buttons()
        player_1_control = Simple_AI_control(ball, player_1)
        player_2_control = Deep_RL_control(*self.shape_of_resize_image)
        all_sprites = pygame.sprite.Group()
        all_sprites.add(player_1)
        all_sprites.add(player_2)
        all_sprites.add(ball)
        player_1_group = pygame.sprite.Group()
        player_1_group.add(player_1)
        player_2_group = pygame.sprite.Group()
        player_2_group.add(player_2)

        pygame.init()
        screen = pygame.display.set_mode((self.width_of_window, self.height_of_window))
        pygame.display.set_caption("Pong")
        clock = pygame.time.Clock()

        screen.fill(WHITE)
        all_sprites.draw(screen)
        pygame.display.flip()

        eps = 0
        MAX_EPS = 20000
        self.frames_limit = self.checkpoints_bias
        self.frames_after_checkpoint = self.frames_limit


        screenshot_1 = screenshot(screen, (self.width_of_window, self.height_of_window))
        screenshot_1 = resize(screenshot_1,
                              self.shape_of_resize_image, preserve_range=True)
        running = True
        while running:
            if self.with_saves and not self.is_checkpoint_loaded:
                self.save(ball, player_1, player_2)


            clock.tick(FPS)
            screenshot_2 = resize(screenshot(screen, (self.width_of_window, self.height_of_window)),
                           self.shape_of_resize_image, preserve_range=True)
            delta_frame_1 = screenshot_2 - screenshot_1



            for event in pygame.event.get():
                if event.type == pygame.KEYUP:
                    buttons_control.KEYUP = True
                    buttons_control.KEYDOWN = False
                if event.type == pygame.KEYDOWN:
                    buttons_control.KEYUP = False
                    buttons_control.KEYDOWN = True
                    if event.key == pygame.K_UP:
                        buttons_control.UP = True
                        buttons_control.DOWN = False
                    elif event.key == pygame.K_DOWN:
                        buttons_control.UP = False
                        buttons_control.DOWN = True
                if event.type == pygame.QUIT:
                    running = False

            command_1 = player_1_control.get_command()
            if random.random() > eps/MAX_EPS:
                command_2 = random.randint(1, 2)
            else:
                command_2 = player_2_control.get_command(delta_frame_1)
            player_1.update(self.height_of_window, self.width_of_window, command=command_1)
            player_2.update(self.height_of_window, self.width_of_window, command=command_2)

            is_collide_with_wall = ball.update(self.height_of_window, self.width_of_window,
                                     x_speed=self.ball_speed_x,
                                     y_speed=self.ball_speed_y)
            if is_collide_with_wall:
                self.ball_speed_y *= -1
            self.collide_upldate(player_1_group, ball, command_1)
            self.collide_upldate(player_2_group, ball, command_2)

            r, is_done = self.ball_final_coord(ball, player_1, player_2)

            if self.frames_after_checkpoint < 0:
                if r == 0:
                    r = 0.01
                    self.saves = []
                self.frames_limit = self.checkpoints_bias
                self.frames_after_checkpoint = self.frames_limit
                self.is_checkpoint_loaded = False
                is_done = True

            screen.fill(WHITE)
            all_sprites.draw(screen)
            pygame.display.flip()
            screenshot_1 = deepcopy(screenshot_2)
            player_2_control.get_command_for_fit(delta_frame_1, r,
                                                 is_done, command_2-1, self.is_checkpoint_loaded,
                                                 self.frames_after_checkpoint)
            if self.is_checkpoint_loaded:
                self.frames_after_checkpoint += -1
            if r != 0:
                eps += 1
                player_2_control.fit(self.is_checkpoint_loaded, saves_len=len(self.saves))
                if r < 0 and len(self.saves) != 0:
                    self.load_checkpoint(ball, player_1, player_2)
                elif r < 0 and len(self.saves) == 0:
                    self.reset(ball, player_1, player_2)
                self.frames_after_checkpoint = self.frames_limit
                if eps % 100 == 0:
                    player_2_control.model.save("model_" + str(eps) + ".h5")

        pygame.quit()


pong = Pong(ball_speed=3,
            player_1_speed=1,
            player_2_speed=1,
            width_of_window=400,
            height_of_window=400,
            with_checkpoints=False,
            checkpoints_bias=10)
pong.main()


