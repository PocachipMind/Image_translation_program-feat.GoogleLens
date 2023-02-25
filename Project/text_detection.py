import cv2
import numpy as np
import torch
from utils.CRAFT.craft import CRAFT
from collections import OrderedDict
import torch.backends.cudnn as cudnn
from utils.CRAFT import craft_utils
from utils.CRAFT import imgproc
from torch.autograd import Variable
from utils.CRAFT import file_utils
import os

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

# CRAFT 모델 로드
net = CRAFT()     # initialize

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
trained_model = "./models/CRAFT/craft_mlt_25k.pth"

print('Loading weights from checkpoint (' + trained_model + ')')
if device == 'cuda':
    net.load_state_dict(copyStateDict(torch.load(trained_model)))
else:
    net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))

if device == 'cuda':
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False


net.eval()







# LinkRefiner ( CRAFT 모델 실행용 )
refine_net = None
refine = True
refiner_model = './models/CRAFT/craft_refiner_CTW1500.pth'
if refine:
    from utils.CRAFT.refinenet import RefineNet
    refine_net = RefineNet()
    print('Loading weights of refiner from checkpoint (' + refiner_model + ')')
    if device == 'cuda':
        refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
        refine_net = refine_net.cuda()
        refine_net = torch.nn.DataParallel(refine_net)
    else:
        refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location='cpu')))

    refine_net.eval()
    poly = True




# CRAFT 모델을 통해 사각형 그리기
def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):

    canvas_size = 1280
    mag_ratio = 1.5

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)


    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text




# 넘파이 값으로 이미지를 입력받으면 원본 이미지에서의 위치와 Numpy값 이미지를 리턴해줌
def Text_Detection(input, device , poly, refine_net):
    
    img = input           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    image = img
    text_threshold, link_threshold, low_text = 0.7, 0.4, 0.4
    bboxes, polys, score_text = test_net(net, image, text_threshold, link_threshold, low_text, device == 'cudea', poly, refine_net)
    # score_text -> 특이점 추출 이미지 cv2.imwrite(mask_file, score_text)로 저장

    
    img = np.array(image[:,:,::-1])

    returnList = []

    # 추출한 polys들을 순회
    for box in polys:
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)

        min_x = 9999999
        min_y = 9999999
        max_x = 0
        max_y = 0
        for j in poly:
            if j[0] <= min_x :
                min_x = j[0]
            if j[0] >= max_x :
                max_x = j[0]
            if j[1] <= min_y :
                min_y = j[1]
            if j[1] >= max_y :
                max_y = j[1]
        
        width = max_x-min_x
        hight = max_y-min_y
        
        if width >= hight*1.3 :
            returnList.append((img[min_y:max_y,min_x:max_x],(min_x,min_y),(max_x,max_y),"original"))
            returnList.append((cv2.rotate(img[min_y:max_y,min_x:max_x], cv2.ROTATE_180),(min_x,min_y),(max_x,max_y),"rotate_180"))
        else:
            returnList.append((cv2.rotate(img[min_y:max_y,min_x:max_x], cv2.ROTATE_90_CLOCKWISE),(min_x,min_y),(max_x,max_y),"rotate_90_clock"))
            returnList.append((cv2.rotate(img[min_y:max_y,min_x:max_x], cv2.ROTATE_90_COUNTERCLOCKWISE),(min_x,min_y),(max_x,max_y),"rotate_90_counterclock"))
            
    
    return returnList
    # 리턴 포맷 -> 이미지, 이미지2, 시작, 끝






if __name__ == "__main__" :

    # test = Text_Detection(cv2.imread('./using_image/input.jpg'),device , poly, refine_net)
    # for i in test:
    #     cv2.imshow("1",i[0])
    #     cv2.waitKey(0)       
    #     print(i)
        
    # Load the input image
    img = './using_image/input.jpg'
    image = imgproc.loadImage(img)

    text_threshold, link_threshold, low_text = 0.7, 0.4, 0.4
    bboxes, polys, score_text = test_net(net, image, text_threshold, link_threshold, low_text, device == 'cudea', poly, refine_net)
    # score_text -> 특이점 추출 이미지 cv2.imwrite(mask_file, score_text)로 저장







    # 이미지 잘라서 저장하기

    # 저장 폴더
    save_folder = './save_detecting/'
    os.makedirs(save_folder, exist_ok=True)
    img = np.array(image[:,:,::-1])

    # 추출한 polys들을 순회
    for i, box in enumerate(polys):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)

        min_x = 9999999
        min_y = 9999999
        max_x = 0
        max_y = 0
        for j in poly:
            if j[0] <= min_x :
                min_x = j[0]
            if j[0] >= max_x :
                max_x = j[0]
            if j[1] <= min_y :
                min_y = j[1]
            if j[1] >= max_y :
                max_y = j[1]
        # print(poly)
        # print(min_x,min_y)
        # print(max_x,max_y)
        # cv2.rectangle(img,(min_x,min_y) ,(max_x,max_y), color=(0, 0, 255), thickness=3)
        # cv2.imshow("d",img[min_y:max_y,min_x:max_x])
        # cv2.waitKey(0)
        cv2.imwrite(save_folder+str(i)+".jpg", img[min_y:max_y,min_x:max_x])

    # Save result image
    #cv2.imshow(img)
    #cv2.waitKey(0)


