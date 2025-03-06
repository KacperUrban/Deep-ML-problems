import numpy as np

def pos_encoding(position: int, d_model: int):
  # Your code here
  if position == 0 or d_model <= 0:
    return -1

  position_ids = np.arange(position).reshape(position, 1)
  dimension_ids = np.arange(d_model).reshape(1, d_model)
  angle_rates = 1 / np.power(10000, (2 * (dimension_ids // 2)) / d_model)
  pos_encoding = position_ids * angle_rates

  pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
  pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
  return pos_encoding.astype(np.float16)