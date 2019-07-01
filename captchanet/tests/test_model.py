import captchanet

def test_build_model():
  image_shape = (400, 120, 3)
  image_type = 'float32'
  vocab_size = 50
  max_len_word = 10
  model = captchanet.build_model(image_shape, image_type, vocab_size, max_len_word)
  assert model.count_params() == 10760764
