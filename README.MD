1. Добавил большой календарь (налоговые, праздничные, предпраздничные дни, дни недели, номера месяцев, номера кварталов)
2. В экзогенные признаки внес медиану и std, детекцию аномалий по лагам
3. Переписал трансформер,чтобы в prediction_mode можно было получить признаки для дня следующего за последним тренировочным, а также добавить новое значение
4. Немного поправил feature_selection, там не работали 2, 3 методы
5. Переписал predict модели, теперь есть 2 функции: predict_next_day, predict_period, отлавливаются аномалии и вслучае чего выдается ошибка

- Кажется арима не работает, для нее надо написать свою model.prepare()
- Запускал крос-валидацию и подбор гиперпараметров, еще считается, но кажется к MAE 0.33 не сойдется ни при каком окне. Поэтому хз, с этим делать
