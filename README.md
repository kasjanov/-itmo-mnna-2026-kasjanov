# Лабораторная работа №2

## Тема

Обучение GPT-like языковой модели на PyTorch и Lightning.

## Архитектура модели

Реализована decoder-only GPT-like модель:

- token embedding;
- sinusoidal positional encoding;
- stack из Transformer decoder blocks;
- multi-head masked self-attention;
- FFN с GELU;
- post-norm LayerNorm;
- LM-head для получения логитов.

В модели используется post-norm вариант:

z1 = LayerNorm(x + Attention(x))

z2 = LayerNorm(z1 + FFN(z1))

Сохранить лучший чекпоинт модели и загрузить его на GitHub (или
на облачный диск):

https://disk.yandex.ru/d/wsnkF9c3a9dEKA

https://disk.yandex.ru/d/ri0wJUkgE4pn-A
<img width="902" height="739" alt="изображение" src="https://github.com/user-attachments/assets/e60c0ee5-e0e5-4cde-9797-6f66d3c6195c" />
