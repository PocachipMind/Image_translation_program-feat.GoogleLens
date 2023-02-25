import numpy as np
import gradio as gr
import torch
import cv2

import text_detection
import text_recognition

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# sampleList = None # 이미지, 첫포인트,끝포인트 양식으로 리스트
cutList = None # test는 잘린 이미지 위치와 변형관련, 나온 text, 점수 이렇게 되있음.
translated_text_list = None # 번역된 텍스트 모음집
changed_image = None # 변형적용된 이미지

# 번역해서 적용해서 보여주기
def show_image(image):

    global cutList
    global translated_text_list
    global changed_image
    
    # 이미지로부터 번역 데이터 리스트 갱신
    sampleList = text_detection.Text_Detection(image ,device , text_detection.poly, text_detection.refine_net)
    cutList = text_recognition.Text_Recognition(text_recognition.replace_opt,sampleList)
    
    # 번역 적용해서 이미지 리턴
    changed_image, cutList, translated_text_list = text_recognition.show_image(image,cutList)

    return changed_image



# 번역된 이미지 정보 불러오기파트
# 1. 입력 가능 범위 출력
def print_info_labels():
    
    global translated_text_list

    if cutList is None or translated_text_list is None:
        return "No Image! Put the Image"
    
    return f"입력값 : 0 ~ {len(translated_text_list)-1}"

# 2. 원본 이미지에 영역표시
def print_info_detection_Image(image):
    
    global cutList

    if cutList is None or translated_text_list is None:
        raise gr.Error("먼저 이미지를 넣어주세요") 
    
    for i in cutList: # [((522, 47), (731, 109), 'original'), 'JamesStewan', tensor(0.1951)]
        cv2.rectangle(image, i[0][0], i[0][1], (0,255,0) ,3)

    return image

# 3. 정보 출력
def print_info_detection_Image_part(detection_num,original_image):
    
    global changed_image
    global cutList # [((522, 47), (731, 109), 'original'), 'JamesStewan', tensor(0.1951)]
    global translated_text_list

    if cutList is None or translated_text_list is None :
        raise gr.Error("이미지를 넣은 이후 입력해주세요.") 

    if not detection_num.isnumeric():
        raise gr.Error("정수로 변환 가능하도록 입력해 주세요.") 

    detection_num = int(detection_num)
    
    if len(translated_text_list) <= detection_num or detection_num < 0:
        raise gr.Error("입력값 안의 정수로 입력해주세요.") # 사실 구조적으로 안나오는게 맞음. 
    
    detection_image_part = original_image[cutList[detection_num][0][0][1]:cutList[detection_num][0][1][1],cutList[detection_num][0][0][0]:cutList[detection_num][0][1][0]]
    detection_image_part_trans = changed_image[cutList[detection_num][0][0][1]:cutList[detection_num][0][1][1],cutList[detection_num][0][0][0]:cutList[detection_num][0][1][0]]
    Detected_text = cutList[detection_num][1]
    teansformed_text = translated_text_list[detection_num]
    confidence_score = f"{cutList[detection_num][2].item()*100:.2f}%"

    return detection_image_part, detection_image_part_trans, Detected_text, teansformed_text, confidence_score



# 편의성을 위한 증감
def increase(num):
    global translated_text_list
    if not num.isnumeric():
        return 0
    num = int(num)
    if num >= len(translated_text_list)-1 or num < 0:
        return 0
    return num + 1
def decrease(num):
    global translated_text_list
    if not num.isnumeric():
        return len(translated_text_list)-1
    num = int(num)
    if num <= 0 or num > len(translated_text_list)-1 :
        return len(translated_text_list)-1
    return num - 1


# 사진찍으면 멈추기
def stop(inp):
    return np.fliplr(inp)


# 구체적 화면 코드 

with gr.Blocks() as demo:

    gr.Markdown("# 이미지 자동 번역 프로그램 (Our Model Version)")
    gr.Markdown("편리하게 이미지를 번역해보세요!")
    gr.HTML("""<div style="display: inline-block;  float: right;">Made By 5 Team : 이성규, 김민정, 이승현, 이주형, 민안세</div>""")

    # 1 번탭
    with gr.Tab("Image Upload"):
        with gr.Row():
            image_input = gr.Image(label="Upload IMG")
            image_output = gr.Image(label="Change Image")        

        image_button = gr.Button("TransForm Image")
        image2_button = gr.Button("TransForm Information")

        with gr.Row():
            detection_image = gr.Image(label="detection_image")
            with gr.Blocks():
                detection_num_range = gr.Textbox(label="Predicted Label Range",interactive=False)
                detection_num = gr.Textbox(label="Predicted Label")
                with gr.Row():
                    plus = gr.Button("+")
                    minus = gr.Button("-")
            with gr.Blocks():
                detection_image_part = gr.Image(label="Detection Image")
                detection_image_part_trans = gr.Image(label="Detection Image(Transform)")
                Detected_text = gr.Textbox(label="Detected_text",interactive=False)
                teansformed_text = gr.Textbox(label="teansformed_text",interactive=False)
                confidence_score = gr.Textbox(label="confidence_score",interactive=False)

        image_button.click(show_image,inputs=image_input, outputs=image_output )
        image2_button.click(print_info_labels, outputs=detection_num_range)
        image2_button.click(print_info_detection_Image, inputs=image_input , outputs=detection_image)
        detection_num.change(print_info_detection_Image_part, inputs=[detection_num,image_input], outputs=[detection_image_part, detection_image_part_trans, Detected_text, teansformed_text, confidence_score])
        plus.click(increase, inputs=detection_num, outputs=detection_num)
        minus.click(decrease, inputs=detection_num, outputs=detection_num)

        gr.Examples(
            examples=["./using_image/2.jpg", "./using_image/5.png", "./using_image/6.jpg", "./using_image/7.png", "./using_image/8.jpg" ],
            inputs=image_input,
        )




    # 2번 탭
    with gr.Tab("Using WebCam"):
        with gr.Row():
            image_web = gr.Image(source="webcam", streaming=True, label="Web Cam")
            image_input = gr.Image(label="Upload IMG")
            image_output = gr.Image(label="Change Image")        

        image_button = gr.Button("TransForm Image")
        image2_button = gr.Button("TransForm Information")

        with gr.Row():
            detection_image = gr.Image(label="detection_image")
            with gr.Blocks():
                detection_num_range = gr.Textbox(label="Predicted Label Range",interactive=False)
                detection_num = gr.Textbox(label="Predicted Label")
                with gr.Row():
                    plus = gr.Button("+")
                    minus = gr.Button("-")
            with gr.Blocks():
                detection_image_part = gr.Image(label="Detection Image")
                detection_image_part_trans = gr.Image(label="Detection Image(Transform)")
                Detected_text = gr.Textbox(label="Detected_text",interactive=False)
                teansformed_text = gr.Textbox(label="teansformed_text",interactive=False)
                confidence_score = gr.Textbox(label="confidence_score",interactive=False)

        image_button.click(stop,inputs=image_web, outputs=image_input )
        image_button.click(show_image,inputs=image_web, outputs=image_output )
        image2_button.click(print_info_labels, outputs=detection_num_range)
        image2_button.click(print_info_detection_Image, inputs=image_input , outputs=detection_image)
        detection_num.change(print_info_detection_Image_part, inputs=[detection_num,image_input], outputs=[detection_image_part, detection_image_part_trans, Detected_text, teansformed_text, confidence_score])
        plus.click(increase, inputs=detection_num, outputs=detection_num)
        minus.click(decrease, inputs=detection_num, outputs=detection_num)




demo.launch(share=True)