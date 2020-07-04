from transformers import BertTokenizer
import torch


def testModel(sentence):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    input_ids = []
    attention_masks = []
    encoded_dict = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=64, pad_to_max_length=True,
                                         return_attention_mask=True, return_tensors='pt')
    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'])
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

    print(input_ids)
    print(attention_masks)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    model = torch.load('bertClass_save.pt')
    model.eval()

    b_input_ids = input_ids.to(device)
    b_input_mask = attention_masks.to(device)

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    print(logits)
    if logits[0][0] > logits[0][1]:
        ans = 0
        print("该语句的语法 错误")
    else:
        ans = 1
        print("该语句的语法 正确")
    return ans


if __name__ == '__main__':
    # sentence = "I am a good man."
    # sentence = "what what good."
    sentence = "100 is bigger than 9954562153. I know it so clearly."

    # sentence = "9954562153"

    print(sentence)
    testModel(sentence)

