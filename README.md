# wildfire-ai
The history of 3rd place in wildfire ai competition

# 0.9305 public leaderboard, 0.9246 private leaderboard, 3rd place

Состоит из двух частей:
* ноутбук содержит исследование и сохранение модели
* директория best_solution содержит распакованный без изменений архив, который был отправлен в систему. Я его совсем не менял (там даже что-то лишнее закинуто), чтобы была полная воспроизводимость.

# Что нужно сделать для запуска:
* скачать из данных NCEP:  air.[year].nc,rhum.[year].nc,uwnd.[year].nc,vwnd.[year].nc для [year] от 2012 до 2019 годов (всего 32 файла): ftp://ftp.cdc.noaa.gov/Datasets/ncep/
* разархивировать модель (лежит в best_solutions, .7z файл)
* скрипт можно запустить на трейне, который лежит здесь (wildfires_train.csv)

