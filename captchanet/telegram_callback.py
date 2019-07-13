from keras.callbacks import Callback


class TelegramCallback(Callback):

    def __init__(self, config, name=None):
        super().__init__()
        
        self.check_telegram_module()
        import telegram
        
        self.user_id = config['telegram_id']
        self.bot = telegram.Bot(config['token'])
        if name != None:
            self.name = name
        else:
            self.name = self.model.name

    def send_message(self, text):
        try:
            self.bot.send_message(chat_id=self.user_id, text=text)
        except Exception as e:
            print('Message did not send. Error: {}.'.format(e))

    def on_train_begin(self, logs={}):
        text = 'Start training model {}.'.format(self.name)
        self.send_message(text)

    def on_train_end(self, logs={}):
        text = 'Training model {} ended.'.format(self.name)
        self.send_message(text)

    def on_epoch_end(self, epoch, logs={}):
        text = '{}: Epoch {}.\n'.format(self.name, epoch)
        for k, v in logs.items():
            if k != "lr":
                text += '{}: {:.4f}; '.format(k, v)
            else:
                text += '{}: {:.6f}; '.format(k, v) #4 decimal places too short for learning rate
        self.send_message(text)
        