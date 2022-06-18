from lab8.PEGASOS import *
from lab8.SGD import *


class User:
    def userAnswer(self):
        """
        Функция создана дл упрощения работы пользователя с данной программой, тут представлены подсказки и премеры ввода
        данных.
        Returns
        ===========
        Обращается к нужной функции метода и передает ей необходимые параметры.
        """
        print(
            "Каким методом хотите воспользоваться?\n"
            "1 - Функцию, решающую задачу поиска экстремума функции методом стохастического градиентного спуска "
            "(SGD). \n"
            "2 - Функцию, реализующую модель классификации на два класса методом опорных векторов SVM с \n"
            "применением алгоритма градиентного спуска для минимизации функции ошибок (PEGASOS algorithm). \n ")

        user_answer = int(input())

        # PEGASOS algorithm
        if user_answer == 2:
            print("Введите параметры для функции make_blobs которая генерирует входные данные")
            print("Введите параметр n_samples. Например 50.")
            n_samples_vvod = int(input())
            print("Введите параметр centers. Например 2.")
            centers_vvod = int(input())
            print("Введите параметр n_features. Например 2.")
            n_features_vvod = int(input())

            X, Y = make_blobs(n_samples=n_samples_vvod, centers=centers_vvod,
                              n_features=n_features_vvod)  # , cluster_std=1.2 )

            for i, j in enumerate(Y):
                if j == 0:
                    Y[i] = -1
                elif j == 1:
                    Y[i] = 1

            X_train = X[len(X) // 5:]
            y_train = Y[len(X) // 5:]

            # training sets
            X_test = X[:len(X) // 5]
            y_test = Y[:len(X) // 5]

            pegasos.SVM(X_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)

        # SGD
        if user_answer == 1:
            print("Демонстрация работы на примере функции -cos(x) * cos(y) * exp(-((x - pi) ** 2 + (y - pi) ** 2))")
            opt_res, f_opt = SGD.stochastic_gradient_descent(max_epochs=500, xy_start=np.array([2.5, 2.5]))
            print(opt_res[-1])
            SGD.SGD_visualize(opt_res)

        else:
            print('Введен неверный номер')


# functionss = User()
# functionss.userAnswer()
