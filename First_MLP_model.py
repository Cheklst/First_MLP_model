import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

#Загрузка данных
data_df = pd.read_csv("data/House_Rent_Dataset.csv")

def describe_data(data_df):
  describe = str(data_df.info())
  describe += "\n" + "-" * 200 + "\n"
  describe += str(data_df.describe())
  describe += "\n" + "-" * 200 + "\n"
  describe += str(data_df.head())
  describe += "\n" + "-" * 200 + "\n"
  describe += str(data_df.isnull().sum())
  describe += "\n" + "-" * 200 + "\n"
  return describe

#Визуальный анализ изначальных данных
print(describe_data(data_df))

#Отброс ненужных параметров
data_df.drop("Posted On", axis=1, inplace=True)
data_df.drop("Tenant Preferred", axis=1, inplace=True)
data_df.drop("Point of Contact", axis=1, inplace=True)

#Разбиение одного параметра на два
data_df[["floor_number", "total_floor"]] = data_df["Floor"].str.lower().str.split("out of", expand=True)
data_df["floor_number"] = data_df["floor_number"].str.strip()
data_df["total_floor"] = data_df["total_floor"].str.strip()

#Замена строковых данных на числовые
data_df["floor_number"] = data_df["floor_number"].replace({
    "ground": 0,
    "lower basement": -2,
    "upper basement": -1
})

#Визуальное сопоставление разделенных данных
print("Разделенные части:")
print(data_df[["Floor", "floor_number", "total_floor"]].head(10))
print("\n" + "-" * 200)
data_df["total_floor"] = data_df["total_floor"].fillna(1)

#Отброс изначального параметра
data_df.drop("Floor", axis=1, inplace=True)

#Визуальный анализ данных
print(describe_data(data_df))

#Выделение целевого признака
X = data_df.drop("Rent", axis=1)
y = data_df["Rent"]

#Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

#Информация по разделению данных
print("Исходный размер данных:", len(X))
print("Тренировочная выборка:", len(X_train), "объектов")
print("Тестовая выборка:", len(X_test), "объектов")
print("\nТренировочные данные:\n"+ str(X_train))
print("\nТестовые данные:\n" + str(X_test))

#Создание DataFrame для визуального анализа обработанных данных
data_df_vizualization = X_train.copy()
data_df_vizualization["Rent"] = y_train.copy()
data_df_vizualization["Rent_log"] = np.log1p(data_df_vizualization["Rent"])
Q1_log = np.percentile(data_df_vizualization["Rent_log"], 25)
Q3_log = np.percentile(data_df_vizualization["Rent_log"], 75)
IQR_log = Q3_log - Q1_log

# Границы для логарифмирования данных
lower_bound_log = Q1_log - 1.5 * IQR_log
upper_bound_log = Q3_log + 1.5 * IQR_log

data_df_vizualization["Rent_log"] = data_df_vizualization["Rent_log"].clip(lower_bound_log, upper_bound_log)


#Общая информация по изначальным данным
print("Информация по изначальным данным:")
print("Количество записей: " + str(len(data_df_vizualization)))
print("Средняя цена: " + str(data_df_vizualization["Rent"].mean()))
print("Медианная цена: " + str(data_df_vizualization["Rent"].median()))
print("Минимальная цена: " + str(data_df_vizualization["Rent"].min()))
print("Максимальная цена: " + str(data_df_vizualization["Rent"].max()))

#Общая информация по модифицированным данным
print("Информация по модифицированным данным:")
print("Количество записей: " + str(len(data_df_vizualization)))
print("Средняя логарифмированная цена: " + str(data_df_vizualization["Rent_log"].mean()))
print("Медианная логарифмированная цена: " + str(data_df_vizualization["Rent_log"].median()))
print("Минимальная логарифмированная цена: " + str(data_df_vizualization["Rent_log"].min()))
print("Максимальная логарифмированная цена: " + str(data_df_vizualization["Rent_log"].max()))


fig, axes = plt.subplots(2, 2, figsize=(12, 8))

#Boxplot Rent(изначальные)
sns.boxplot(y=data_df_vizualization["Rent"],  ax=axes[0,0])
axes[0,0].set_title("Rent (исходные данные)")
axes[0,0].set_ylabel("Rent")

#Boxplot Rent(модифицированные)
sns.boxplot(y=data_df_vizualization["Rent_log"],  ax=axes[0,1])
axes[0,1].set_title("Rent (логарифмированные данные)")
axes[0,1].set_ylabel("log(1 + Rent)")

# Boxplot Rent без выбросов(изначальные)
sns.boxplot(y=data_df_vizualization["Rent"], showfliers=False, ax=axes[1,0])
axes[1,0].set_title("Rent (исходные данные) без выбросов")
axes[1,0].set_ylabel("Rent")

# Boxplot Rent без выбросов(модифицированные)
sns.boxplot(y=data_df_vizualization["Rent_log"], showfliers=False, ax=axes[1,1])
axes[1,1].set_title("Rent_log без выбросов")
axes[1,1].set_ylabel("Rent_log")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 2, figsize=(12, 8))

#График средних цен по городам(изначальные)
city_means = data_df_vizualization.groupby("City")["Rent"].mean()
city_means.plot(kind="bar", ax=axes[0,0])
axes[0,0].set_title("Средняя цена по городам (исходные данные)")

#График средних цен по городам(модифицированные)
city_means = data_df_vizualization.groupby("City")["Rent_log"].mean()
city_means.plot(kind="bar", ax=axes[0,1])
axes[0,1].set_title("Средняя цена по городам (модифицированные данные)")

#Гистограмма распределения цен(изначальные)
sns.histplot(data_df_vizualization["Rent"], kde=True, bins=30, ax=axes[1,0])
axes[1,0].set_title("Распределение цен (исходные данные)")

#Гистограмма распределения цен(модифицированные)
sns.histplot(data_df_vizualization["Rent_log"], kde=True, bins=30, ax=axes[1,1])
axes[1,1].set_title("Распределение цен (модифицированные данные)")

#Тепловая карта корреляции(изначальные)
corr_raw = data_df_vizualization[["Rent", "Size", "BHK", "Bathroom"]].corr()
sns.heatmap(corr_raw, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[2,0])
axes[2,0].set_title("Корреляция (исходные данные)")

#Тепловая карта корреляции (модифицированные)
corr_raw = data_df_vizualization[["Rent_log", "Size", "BHK", "Bathroom"]].corr()
sns.heatmap(corr_raw, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[2,1])
axes[2,1].set_title("Корреляция(модифицированные данные)")

plt.tight_layout()
plt.show()

#Логарифмирование целевой переменной и применение метода IQR
y_train_log = np.log1p(y_train)

Q1_log = np.percentile(y_train_log, 25)
Q3_log = np.percentile(y_train_log, 75)
IQR_log = Q3_log - Q1_log
lower_bound_log = Q1_log - 1.5 * IQR_log
upper_bound_log = Q3_log + 1.5 * IQR_log

y_train_log_clipped = y_train_log.copy()
y_train_log_clipped = y_train_log_clipped.clip(lower_bound_log, upper_bound_log)

#Нормализация
print(f"\nТипы данных:\n{X_train.dtypes}")

numeric_features = ["Size", "Bathroom", "BHK", "total_floor", "floor_number"]
print(f"Числовые признаки: {numeric_features}")

#Выделение числовых признаков
X_train_num = X_train[numeric_features].copy()
X_test_num = X_test[numeric_features].copy()

#Обучение scaler тренировочных данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_num)

#Применение scaler к тестовым данным
X_test_scaled = scaler.transform(X_test_num)

#Преобразование в DataFrame
X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=[f"{col}_scaled" for col in numeric_features],
    index=X_train.index
)

X_test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=[f"{col}_scaled" for col in numeric_features],
    index=X_test.index
)

#Кодирование
categorical_features = ["Area Type", "Furnishing Status", "City", "Area Locality"]
for col in categorical_features:
    unique_vals = X_train[col].nunique()
    print(f"{col}: {unique_vals} уникальных значений")

onehot_features = ["Area Type", "Furnishing Status", "City"]
target_encode_features = ["Area Locality"]
print(f"OneHot: {onehot_features}")
print(f"Target Encoding: {target_encode_features}")

#Применение One-hot
onehot_encoder = OneHotEncoder(
    sparse_output=False,
    drop="first",
    handle_unknown="ignore"
)

X_train_onehot = onehot_encoder.fit_transform(X_train[onehot_features])
X_test_onehot = onehot_encoder.transform(X_test[onehot_features])
onehot_feature_names = onehot_encoder.get_feature_names_out(onehot_features)

#Преобразуем в DataFrame
X_train_onehot_df = pd.DataFrame(
    X_train_onehot,
    columns=onehot_feature_names,
    index=X_train.index
)
X_test_onehot_df = pd.DataFrame(
    X_test_onehot,
    columns=onehot_feature_names,
    index=X_test.index
)

#Применение TargetEncoder
target_encoder = TargetEncoder(target_type="continuous", random_state=42)

X_train_target_encoded_df = pd.DataFrame(
    target_encoder.fit_transform(X_train[target_encode_features], y_train_log_clipped),
    columns=target_encode_features,
    index=X_train.index
)

X_test_target_encoded_df = pd.DataFrame(
    target_encoder.transform(X_test[target_encode_features]),
    columns=target_encode_features,
    index=X_test.index
)

#Объединение всех данные в DataFrame
X_train_final = pd.concat([
    X_train_scaled_df,
    X_train_target_encoded_df,
    X_train_onehot_df
], axis=1)

X_test_final = pd.concat([
    X_test_scaled_df,
    X_test_target_encoded_df,
    X_test_onehot_df
], axis=1)

print(f"X_train_final: {X_train_final.shape}")
print(f"X_test_final: {X_test_final.shape}")

X_train_array = X_train_final.values.astype("float32")
X_test_array = X_test_final.values.astype("float32")
y_train_array = y_train_log_clipped.values.astype("float32")
y_test_array = y_test.values.astype("float32")

#Создание модели
model = keras.Sequential([
    #Входной слой
    layers.Input(shape=(X_train_array.shape[1],)),

    #Первый скрытый слой
    layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
    layers.Dropout(0.3, seed=42),

    #Второй скрытый слой
    layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
    layers.Dropout(0.3, seed=42),

    # Выходной слой
    layers.Dense(1, activation="linear")
])

#Вывод архитектуры модели
model.summary()

#Компиляция модель
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae", "mse"]
)

#Обучение модели
history = model.fit(
    X_train_array, y_train_array,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    shuffle=True,
    verbose=1
)

#Предсказания модели
y_pred_log_basic = model.predict(X_test_array).flatten()
y_pred_basic = np.expm1(y_pred_log_basic)

#Оценка модели
mae = mean_absolute_error(y_test_array, y_pred_basic)
rmse = np.sqrt(mean_squared_error(y_test_array, y_pred_basic))
r2 = r2_score(y_test_array, y_pred_basic)

print("Результаты модели:")
print("MAE: " + str(round(mae, 2)))
print("RMSE: " + str(round(rmse, 2)))
print("R²: " + str(round(r2, 2)))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#График Loss
axes[0].plot(history.history["loss"], label="Потери train")
axes[0].plot(history.history["val_loss"], label="Потери validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("Эпоха")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

#График MAE
axes[1].plot(history.history["mae"], label="MAE train")
axes[1].plot(history.history["val_mae"], label="MAE validation")
axes[1].set_title("MAE")
axes[1].set_xlabel("Эпоха")
axes[1].set_ylabel("MAE")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

#График сравнения предсказаний и фактических значений цены
plt.figure(figsize=(12, 6))
plt.plot(y_test_array[:100], 'b-', label="Факт")
plt.plot(y_pred_basic[:100], 'r--', label="Прогноз")
plt.ylabel("Rent")
plt.xlabel("Номер примера")
plt.title("Фактические и прогнозируемые значения")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#Создание улучшенной модели
model_adv = keras.Sequential([
    #Входной слой
    layers.Input(shape=(X_train_array.shape[1],)),
    layers.BatchNormalization(),

    #Первый скрытый слой
    layers.Dense(128,  kernel_initializer="he_normal"),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.2),
    layers.Dropout(0.3, seed=42),

    #Второй скрытый слой
    layers.Dense(64, kernel_initializer="he_normal"),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.2),
    layers.Dropout(0.3, seed=42),

    # Выходной слой
    layers.Dense(1, activation="linear")
])

#Вывод архитектуры модели
model_adv.summary()

#Компиляция модель
model_adv.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae", "mse"]
)

#Обучение модели
history_adv = model_adv.fit(
    X_train_array, y_train_array,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    shuffle=True,
    verbose=1
)

#Предсказания модели
y_pred_log_basic_adv = model_adv.predict(X_test_array).flatten()
y_pred_basic_adv = np.expm1(y_pred_log_basic_adv)

#Оценка модели
mae_adv = mean_absolute_error(y_test_array, y_pred_basic_adv)
rmse_adv = np.sqrt(mean_squared_error(y_test_array, y_pred_basic_adv))
r2_adv = r2_score(y_test_array, y_pred_basic_adv)

print("Результаты модели:")
print("MAE: " + str(round(mae_adv, 2)))
print("RMSE: " + str(round(rmse_adv, 2)))
print("R²: " + str(round(r2_adv, 2)))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#График Loss
axes[0].plot(history_adv.history["loss"], label="Потери train")
axes[0].plot(history_adv.history["val_loss"], label="Потери validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("Эпоха")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

#График MAE
axes[1].plot(history_adv.history["mae"], label="MAE train")
axes[1].plot(history_adv.history["val_mae"], label="MAE validation")
axes[1].set_title("MAE")
axes[1].set_xlabel("Эпоха")
axes[1].set_ylabel("MAE")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

#График сравнения предсказаний и фактических значений цены
plt.figure(figsize=(12, 6))
plt.plot(y_test_array[:100], 'b-', label="Факт")
plt.plot(y_pred_basic_adv[:100], 'r--', label="Прогноз базовой модели")
plt.plot(y_pred_basic[:100], 'g--', label="Прогноз улучшенной модели")
plt.ylabel("Rent")
plt.xlabel("Номер примера")
plt.title("Фактические и прогнозируемые значения")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()