# 导入必要的库
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# 随机种子用于保证实验的可重复性
from numpy.random import seed
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # 设置TensorFlow的日志记录级别为ERROR
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

# 设置随机种子，保证实验结果的可重复性
seed(10)
tf.random.set_seed(10)

# 加载、平均并合并传感器样本数据
data_dir = r'bearing_data'  # 数据目录
merged_data = pd.DataFrame()
for filename in os.listdir(data_dir):
    if filename == ".DS_Store":
        continue
    dataset = pd.read_csv(os.path.join(data_dir, filename), encoding='utf-8', sep='\t')
    dataset_mean_abs = np.array(dataset.abs().mean())  # 计算每个文件中数据的绝对值平均值
    dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1, 4))  # 重塑数据为1行4列的DataFrame
    dataset_mean_abs.index = [filename]  # 使用文件名作为索引
    merged_data = merged_data._append(dataset_mean_abs)

merged_data.columns = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']  # 设置列名

# 将数据文件索引转换为日期时间格式并按时间顺序排序
merged_data.index = pd.to_datetime(merged_data.index, format='%Y.%m.%d.%H.%M.%S')
merged_data = merged_data.sort_index()
merged_data.to_csv('Averaged_BearingTest_Dataset.csv')  # 保存合并后的数据为CSV文件
print("Dataset shape:", merged_data.shape)
merged_data.head()

# 划分训练集和测试集
train = merged_data['2004-02-12 10:52:39': '2004-02-15 12:52:39']
test = merged_data['2004-02-15 12:52:39':]
print("Training dataset shape:", train.shape)
print("Test dataset shape:", test.shape)

# 绘制训练集数据的时间序列图
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(train['Bearing 1'], label='Bearing 1', color='blue',  linewidth=1)
ax.plot(train['Bearing 2'], label='Bearing 2', color='red',  linewidth=1)
ax.plot(train['Bearing 3'], label='Bearing 3', color='green',  linewidth=1)
ax.plot(train['Bearing 4'], label='Bearing 4', color='black',  linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Bearing Sensor Training Data', fontsize=16)
plt.show()

# 使用快速傅立叶变换将数据从时域转换到频域
print(train)
train_fft = np.fft.fft(train)
print(train_fft)
test_fft = np.fft.fft(test)

# 绘制频域图
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(train_fft[:, 0].real, label='Bearing 1', color='blue',  linewidth=1)
ax.plot(train_fft[:, 1].imag, label='Bearing 2', color='red',  linewidth=1)
ax.plot(train_fft[:, 2].real, label='Bearing 3', color='green',  linewidth=1)
ax.plot(train_fft[:, 3].real, label='Bearing 4', color='black',  linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Bearing Sensor Training Frequency Data', fontsize=16)
plt.show()

# 绘制测试数据的频域图
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(test_fft[:, 0].real, label='Bearing 1', color='blue',  linewidth=1)
ax.plot(test_fft[:, 1].imag, label='Bearing 2', color='red', linewidth=1)
ax.plot(test_fft[:, 2].real, label='Bearing 3', color='green',  linewidth=1)
ax.plot(test_fft[:, 3].real, label='Bearing 4', color='black', linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Bearing Sensor Test Frequency Data', fontsize=16)
plt.show()

# 数据归一化
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)
scaler_filename = "scaler_data"
joblib.dump(scaler, scaler_filename)  # 保存归一化器

# 重塑输入数据以适应LSTM模型的输入格式 [样本数, 时间步长, 特征数]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
print(X_train)
print("Training data shape:", X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print("Test data shape:", X_test.shape)

# 定义自动编码器模型
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True,
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)  # 使用RepeatVector将编码器的输出复制N份作为解码器的输入
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)
    model = Model(inputs=inputs, outputs=output)
    return model

# 创建自动编码器模型
model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')  # 使用adam优化器和平均绝对误差作为损失函数
print(model.summary())  # 输出模型结构


# 训练模型
nb_epochs = 100
batch_size = 10
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.05).history

# 绘制训练损失曲线
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()

# 绘制训练集的损失分布
X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=train.columns)
X_pred.index = train.index

scored = pd.DataFrame(index=train.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred - Xtrain), axis=1)
plt.figure(figsize=(16, 9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
sns.distplot(scored['Loss_mae'], bins=20, kde=True, color='blue')
plt.xlim([0.0, .5])
plt.show()

# 计算测试集的损失
X_pred = model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=test.columns)
X_pred.index = test.index

scored = pd.DataFrame(index=test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred - Xtest), axis=1)
scored['Threshold'] = 0.275  # 设置阈值，用于区分正常与异常数据
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
print(scored.head())

# 计算训练集的相同指标并合并所有数据以便绘图
X_pred_train = model.predict(X_train)
X_pred_train = X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[2])
X_pred_train = pd.DataFrame(X_pred_train, columns=train.columns)
X_pred_train.index = train.index

scored_train = pd.DataFrame(index=train.index)
scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train - Xtrain), axis=1)

"""
基于统计特性：
根据训练集损失的统计特性来设置阈值，例如使用平均值（mean）和标准差（standard deviation），或使用百分位数（percentile）。

mean_loss = np.mean(scored_train['Loss_mae'])
std_loss = np.std(scored_train['Loss_mae'])
threshold = mean_loss + 3 * std_loss  # 使用3个标准差作为阈值

百分位数法：
选择损失的某个百分位数作为阈值。


threshold = np.percentile(scored_train['Loss_mae'], 95)  # 使用95百分位数作为阈值


改进代码，自动计算阈值
使用上述方法之一来改进代码，自动计算阈值，而不是使用固定值0.275。例如，使用百分位数法：

X_pred_train = model.predict(X_train)
X_pred_train = X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[2])
X_pred_train = pd.DataFrame(X_pred_train, columns=train.columns)
X_pred_train.index = train.index

scored_train = pd.DataFrame(index=train.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train - Xtrain), axis=1)

# 使用95百分位数作为阈值
threshold = np.percentile(scored_train['Loss_mae'], 95)
scored_train['Threshold'] = threshold
scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
scored = pd.concat([scored_train, scored])

print(f"Calculated threshold: {threshold}")

"""



scored_train['Threshold'] = 0.275
scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
scored = pd.concat([scored_train, scored])

# 绘制轴承故障时间图
scored.plot(logy=True, figsize=(16, 9), ylim=[1e-2, 1e2], color=['blue', 'red'])
plt.show()

# 以h5格式保存所有模型信息，包括权重
model.save("Cloud_model.h5")
print("Model saved")




