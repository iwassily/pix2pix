# pix2pix
В этом репозитории моя имплементация обусловленной генеративно-состязательной сети pix2pix из статьи ["Image-to-Image Translation with Conditional Adversarial Networks"](https://arxiv.org/abs/1611.07004), сделанная в рамках прохождения курса ["Deep Learning, семестр 1"](https://stepik.org/course/91157/).

В статье рассмотрена единая архитектура, решающая много различных задач на наборах связанных изображений: раскраска черно-белых фотографий, генерация фотографии по контурам, перевод ночного пейзажа в дневной и т.д.

Для обучения и проведения экспериментов я реализовал класс pix2pix_trainer, его методы позволяют обучать и дообучать модель, сохранять и загружать веса, выводить сгенерированные изображения, историю изменения лоссов. Пример работы с ним есть блокноте [pokemon_colorization.ipynb](https://github.com/iwassily/pix2pix/blob/main/pokemon_colorization.ipynb).
```python
from trainer import pix2pix_trainer
from dataloader import make_dataloaders
train_loader, validation_loader = make_dataloaders(tr_path='datasets/pokemon/tr/', val_path='datasets/pokemon/val')
trainer = pix2pix_trainer(tr_loader=train_loader, val_loader=validation_loader, checkpoint_path='models/pokemon')
trainer.load_checkpoint()
trainer.show_examples(num_examples=4, mode='val')
```

Я запускал свою модель на двух датасетах. Первый это Cityscapes dataset, рассмотренный в оригинальной статье, решается задача восстановления фотографии по её семантической сегментации.
Второй это [Sketch2Pokemon](https://www.kaggle.com/norod78/sketch2pokemon), решается задача раскраски чёрно-белых скетчей.
# Cityscapes
![результаты](https://github.com/iwassily/pix2pix/blob/main/examples/cityscapes_train.png)

результаты генерации для трейна

![результаты](https://github.com/iwassily/pix2pix/blob/main/examples/cityscapes_val.png)

результаты генерации для валидации

Модель обучалась 200 эпох, размер батча 8, 2975 изображений в обучении, 5.5 часов на Tesla P100. 
* [архив с датасетом](https://drive.google.com/file/d/1-CB9XEiPjeRFvF_4cXxdhcW1zwT74FiO/view?usp=sharing)
* [архив с весами обученной модели](https://drive.google.com/file/d/1-1BBha6TeSAeZcdi-v-rK28vxGWxwI0P/view?usp=sharing)

# Sketch2Pokemon
![Резкльтаты](https://github.com/iwassily/pix2pix/blob/main/examples/pokemons.png)

Модель обучалась 100 эпох, размер батча 4, 808 изображений в обучении, 3.5 часа на Tesla P100. 
Датасет был составлен не совсем корректно - в трейне присутствовали изображения из теста. Поскольку модель на таком простом датасете теоретически может запоминать тренировочные изображения, я удалил из них тесовые, чтобы можно было адекватно оценивать качество генерации.
* [архив с датасетом](https://drive.google.com/file/d/1-FwGxm_cVvZZ3560PB8NTCFRjpcWZwdK/view?usp=sharing)
* [архив с весами обученной модели](https://drive.google.com/file/d/1D3YY519p3JcYAHxfn82VX1eyio_FiK18/view?usp=sharing)


# Материалы
Код написан самостоятельно, использовались материалы курса, в редких случаях сверялся с приведенными ниже имплементациями 
* ["Deep Learning, семестр 1"](https://stepik.org/course/91157/) от [Deep Learning School ФПМИ МФТИ](https://www.dlschool.org/)
* ["Image-to-Image Translation with Conditional Adversarial Networks"](https://arxiv.org/abs/1611.07004) оригинальная статья
* [Код одного из авторов статьи](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/)
* [Видео с разбором статьи](https://www.youtube.com/watch?v=9SGs4Nm0VR4) и [код автора видео](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/Pix2Pix)
* [Cityscapes Dataset](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/cityscapes.zip)
* [Sketch2Pokemon dataset](https://www.kaggle.com/norod78/sketch2pokemon)
* [Build Your Own PyTorch Trainer](https://www.youtube.com/watch?v=8ua0qfbPnfk) видео про создание удобного трейнера
