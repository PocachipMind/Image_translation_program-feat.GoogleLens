import string
import argparse
import cv2
from PIL import ImageFont,ImageDraw,Image

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from transformers import pipeline

from utils.DTRB.utils import CTCLabelConverter, AttnLabelConverter
from utils.DTRB.dataset import AlignCollate, CustomProjectDataset
from utils.DTRB.model import Model

from transformers import pipeline
pipe = pipeline("translation", model="circulus/kobart-trans-en-ko-v2")

class replace_opt:
    image_folder = "./demo_image/"
    workers = 0
    batch_size = 192
    saved_model = "./models/DTRB/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth"

    # Data  processing
    batch_max_length = 25
    imgH = 32
    imgW = 100
    rgb = False
    character = '0123456789abcdefghijklmnopqrstuvwxyz'
    sensitive = True
    PAD = False

    # Model Architecture
    Transformation = "TPS"
    FeatureExtraction = "ResNet"
    SequenceModeling = "BiLSTM"
    Prediction = "Attn"
    num_fiducial = 20
    input_channel = 1
    output_channel = 512
    hidden_size = 256



""" vocab / character number configuration """
if replace_opt.sensitive:
    replace_opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
cudnn.benchmark = True
cudnn.deterministic = True
replace_opt.num_gpu = torch.cuda.device_count()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Text_Recognition(opt,ImageList):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    
    
    Project_data = CustomProjectDataset(ImageList, opt=opt)  # custom Dataset
    Project_loader = torch.utils.data.DataLoader( 
        Project_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in Project_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)



            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            returnList = []
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                returnList.append([img_name,pred,confidence_score])
    
    return returnList

def show_image(image, cutList):

    h,w , _ = image.shape
    
    select_higer = []
    # Draw_raw_text_list = []
    for i in range(len(cutList)//2):
        higer = cutList[i*2] if cutList[i*2][2] > cutList[i*2+1][2] else cutList[i*2+1]
        if higer[2] <= 0.25:
            continue
        select_higer.append(higer)
        # Draw_raw_text_list.append(higer[1])
        
        # 평균값으로 이미지 채우기
        img_temp = image[higer[0][0][1]:higer[0][1][1],higer[0][0][0]:higer[0][1][0]].copy()
        img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = np.average(image[higer[0][0][1]:higer[0][1][1],higer[0][0][0]:higer[0][1][0]], axis=(0,1))
        
        image[higer[0][0][1]:higer[0][1][1],higer[0][0][0]:higer[0][1][0]]= img_temp

        # print(cutList[i*2][2])
        # print(cutList[i*2+1][2])
        # print(select_higer)
        # cv2.imshow("",image)
        # cv2.waitKey(0)

    # Draw_text_list = generate_text.Translation_text(Draw_raw_text_list)

    font = ImageFont.truetype("./font/SuseongDotum.ttf",20)
    image = Image.fromarray(image)

    translated_text_list = []
    for i in select_higer:
        # Draw_text = i[1]
        Draw_text =pipe(i[1])[0]['translation_text'] # 번역한 것 
        translated_text_list.append(Draw_text)
        # print(Draw_text , i[1])

        # Draw_text = Draw_text_list[i]


        x1,y1 , x2,y2 = i[0][0][0], i[0][0][1], i[0][1][0], i[0][1][1]
        
        # 어캐 바꿧냐에따라 텍스트 입력하기.
        if i[0][2] == "original":
            # cv2.putText(image, Draw_text,(x1,y2-10),cv2.FONT_HERSHEY_SIMPLEX,1,(254,1,15),2,cv2.LINE_AA )
            draw = ImageDraw.Draw(image)
            draw.text(( x1+(x2-x1)//4, y1+(y2-y1)//4 ), Draw_text,(15,15,255), font=font)

        
        elif i[0][2] == "rotate_180":
            image = image.transpose(Image.ROTATE_180)
            x1_new = w - x2
            y1_new = h - y2
            x2_new = w - x1
            y2_new = h - y1

            # cv2.putText(image, Draw_text,(x1_new,y2_new-10),cv2.FONT_HERSHEY_SIMPLEX,1,(254,1,15),2,cv2.LINE_AA )
            draw = ImageDraw.Draw(image)
            draw.text((x1_new+(x2_new-x1_new)//4,y1_new+((y2_new-y1_new)//4)), Draw_text,(15,15,255), font=font)

            image = image.transpose(Image.ROTATE_180)
        
        elif i[0][2] == "rotate_90_clock":
            image = image.transpose(Image.ROTATE_270)
            x1_new = h - y2
            y1_new = x1
            x2_new = h - y1
            y2_new = x2

            
            # cv2.putText(image, Draw_text,(x1_new,y2_new-10),cv2.FONT_HERSHEY_SIMPLEX,1,(254,1,15),2,cv2.LINE_AA )
            draw = ImageDraw.Draw(image)
            draw.text((x1_new+(x2_new-x1_new)//4,y1_new+((y2_new-y1_new)//4)), Draw_text,(15,15,255), font=font)

            image = image.transpose(Image.ROTATE_90)
        
        elif i[0][2] == "rotate_90_counterclock":
            image = image.transpose(Image.ROTATE_90)
            x1_new = y1
            y1_new = w - x2
            x2_new = y2
            y2_new = w - x1

            # cv2.putText(image, Draw_text,(x1_new,y2_new-10),cv2.FONT_HERSHEY_SIMPLEX,1,(254,1,15),2,cv2.LINE_AA )
            draw = ImageDraw.Draw(image)
            draw.text((x1_new+(x2_new-x1_new)//4,y1_new+((y2_new-y1_new)//4)), Draw_text,(15,15,255), font=font)

            image = image.transpose(Image.ROTATE_270)


        # cv2.imshow("",image)
        # cv2.waitKey(0)

    image = np.array(image)

    # cv2.imshow("",image)
    # cv2.waitKey(0)

    return image, select_higer ,translated_text_list
            



if __name__ == "__main__":
    import text_detection
    image_read = cv2.imread('./using_image/2.jpg')
    sampleList = text_detection.Text_Detection(image_read,device , text_detection.poly, text_detection.refine_net)

    # print(sampleList) # 이미지, 첫포인트,끝포인트 양식으로 리스트

    test = Text_Recognition(replace_opt,sampleList) # test는 잘린 이미지, 나온 text, 점수 이렇게 되있음.
    # [((523, 48), (730, 110), 'original'), 'James', tensor(0.0856)] 이런 포맷임
    # for i in test:
    #     print(i[1], i[2])
    #     cv2.imshow(i[1]+" "+f"{i[2].item():.4f}",image_read[i[0][0][1]:i[0][1][1],i[0][0][0]:i[0][1][0]])
    #     cv2.waitKey(0)

    show_image(image_read,test)
    
