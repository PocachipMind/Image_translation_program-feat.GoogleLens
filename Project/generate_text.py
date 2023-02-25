from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("circulus/kobart-trans-en-ko-v2")

model = AutoModelForSeq2SeqLM.from_pretrained("circulus/kobart-trans-en-ko-v2")


def Translation_one_text(src_text): # str 들어오면 str 반환

    translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))


    return tokenizer.decode(translated.squeeze(), skip_special_tokens=True)


def Translation_text(src_text): #리스트 들어오면 리스트 반환 

    translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))

    changed = []
    for t in translated:
        changed.append(tokenizer.decode(t, skip_special_tokens=True) )


    return changed


from transformers import MarianMTModel, MarianTokenizer


print(model("HELP!!"))

# for t in translated:
#     print( tokenizer.decode(t, skip_special_tokens=True) )

# expected output:
#     2, 4, 6 등은 짝수입니다.
#     그래


if __name__ == "__main__":
    from transformers import pipeline
    pipe = pipeline("translation", model="circulus/kobart-trans-en-ko-v2")
    print(pipe("2, 4, 6 etc. are even numbers."))

    