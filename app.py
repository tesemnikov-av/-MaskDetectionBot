import telebot
import torch
import torchvision.models as models
import torch.nn as nn
import cv2
import torchvision.transforms as T
import yaml
from box import Box

with open("config.yml", "r") as ymlfile:
  cfg = Box(yaml.safe_load(ymlfile))

bot = telebot.TeleBot(cfg.base.token)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = models.resnet50(pretrained=False).to(device)
    
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(device)

model.load_state_dict(torch.load(cfg.base.model_weights))

new_transform = T.Compose([T.ToPILImage(), T.Resize((cfg.base.img_size, cfg.base.img_size)), 
                           T.ToTensor()])

def get_predict(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return new_transform(image)

@bot.message_handler(content_types=['photo'])
def send_photo(message):
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)

    with open(img_name, 'wb') as new_file:
        new_file.write(downloaded_file)

    prediction = get_predict(cfg.base.img_name)
    answer = (model(torch.unsqueeze(prediction, 0)).argmax(-1))[0]
    
    if answer:
        sticker = cfg.base.sticker_yes
    else:
        sticker = cfg.base.sticker_no

    bot.send_sticker(message.chat.id, sticker )
    
bot.polling()

