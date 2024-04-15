import os
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Grayscale, Resize
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from datetime import datetime
from attention_unet import AttentionUNet
from resnet import ResNet18
from train_bb import compute_iou

def check_model_const(model_path, input_image, output_bb, show=True, short_print=False):
    """
    Checks trained model output mask and compares with expected mask
    """

    start_time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    results_path = f"."
    os.makedirs(results_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_filename = os.path.splitext(os.path.basename(input_image))[0]
    original = cv2.imread(input_image)
    with open(output_bb, 'r') as f:
        data = f.readline().split()
        box = [int(x) for x in data]

    expected = np.array(box)
    original_resized_img = cv2.resize(original, (256, 256))

    transform = Compose([Resize((256, 256)), ToTensor()])
    original_resized = torch.from_numpy(original_resized_img).float().permute(2, 0, 1).unsqueeze(0).to(device)
    # expected = torch.from_numpy(expected).float().permute(2, 0, 1).unsqueeze(0).to(device)

    checkpoint = torch.load(model_path)

    model = ResNet18().to(device)
    model.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        predicted = model(original_resized)
    # if 'model' in checkpoint:
    #     model = checkpoint['model']
    #     predicted = model(original)
    # elif 'model_state_dict' in checkpoint:
    #     model = AttentionUNet(3, 1)
    #     model.load_state_dict(torch.load(model_path)['model_state_dict'])
    #     model = model.to(device)
    #     model.eval()
    #
    #     with torch.no_grad():
    #         predicted = model(original)
    # else:
    #     print("Something wrong with model! Try another...")
    #     return

    # precision, recall, f1 = ModelTrainer.calculate_metrics(predicted, expected)
    # precision = round(precision * 100, 2)
    # recall = round(recall * 100, 2)
    # f1 = round(f1 * 100, 2)
    # print(f"P = {precision}% | R = {recall}% | F1 = {f1}%")

    # Masking

    # original_data = original.data.cpu().numpy().squeeze()
    # expected_data = expected.data.cpu().numpy().squeeze()
    predicted_data = predicted.data.cpu().numpy().squeeze()
    # predicted_data[0] = predicted_data[0] * 320
    # predicted_data[1] = predicted_data[1] * 285
    # predicted_data[2] = predicted_data[2] * 320
    # predicted_data[3] = predicted_data[3] * 285

    iou = compute_iou(expected, predicted_data)
    print(iou)

    # predicted_data[0] = predicted_data[0] * original.shape[1] * original.shape[1] / 256
    # predicted_data[1] = predicted_data[1] * original.shape[0] * 256 / original.shape[0]
    # predicted_data[2] = predicted_data[2] * original.shape[1] * original.shape[1] / 256
    # predicted_data[3] = predicted_data[3] * original.shape[0] * 256 / original.shape[0]


    # cv2.rectangle(original_resized_img, (int(predicted_data[0]), int(predicted_data[1])), (int(predicted_data[2]), int(predicted_data[3])), (0, 0, 225), 2)
    cv2.rectangle(original, (int(box[0]), int(box[1])),
               (int(box[2]), int(box[3])), (0, 0, 255), 2)

    cv2.rectangle(original, (int(predicted_data[0]), int(predicted_data[1])),
               (int(predicted_data[2]), int(predicted_data[3])), (255, 0, 0), 2)

    output = cv2.resize(original, (original.shape[1], original.shape[0]))
    cv2.imshow('org Image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Resize prediction to match original image size
    pred_resized = cv2.resize(predicted_data, (original.shape[1], original.shape[0]))

    # Create a side-by-side comparison
    combined = np.hstack((input_image, cv2.cvtColor(pred_resized, cv2.COLOR_GRAY2BGR)))
    cv2.imshow('Original vs. Predicted', combined)
    cv2.waitKey(0)

    # Convert to grayscale
    # grayscale_image = cv2.cvtColor(predicted_data, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Image', predicted_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # predicted_data_save = Image.fromarray((predicted_data * 255).astype('uint8'))
    # predicted_data_save.save(f"{results_path}/results-mask-{input_filename}.jpg")
    #
    # expected_mask = cv2.normalize(expected_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # predicted_mask = cv2.normalize(predicted_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #
    # threshold1, threshold2, color1, color2, color3 = 127, 127, [0, 0, 255], [255, 0, 0], [0, 255, 0]
    # expected_mask_colored = cv2.cvtColor(expected_mask, cv2.COLOR_BGR2RGB)
    # expected_mask_colored[expected_mask > threshold1] = np.array(color1)
    # predicted_mask_colored = cv2.cvtColor(predicted_mask, cv2.COLOR_BGR2RGB)
    # predicted_mask_colored[predicted_mask > threshold1] = np.array(color2)
    #
    # final_mask = cv2.add(expected_mask_colored, predicted_mask_colored)
    # red_in_expected_mask = np.all(expected_mask_colored == color1, axis=-1)
    # blue_in_predicted_mask = np.all(predicted_mask_colored == color2, axis=-1)
    # # final_mask[np.logical_and(red_in_expected_mask, blue_in_predicted_mask)] = color3
    # #
    # # original_data_masked = cv2.normalize(original_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # # original_data_masked = cv2.cvtColor(original_data_masked, cv2.COLOR_BGR2RGB)
    # # original_data_masked[final_mask > threshold2] = final_mask[final_mask > threshold2]
    #
    # # Plotting
    #
    # titles = ['(a)', '(b)', '(c)', '(d)'] if short_print else ['Original', 'Expected', 'Predicted', 'Comparing']
    # fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    # # axs[0].imshow(original_data)
    # # axs[0].set_title(titles[0], weight='bold')
    # # axs[1].imshow(expected_data)
    # # axs[1].set_title(titles[1], weight='bold')
    # # axs[2].imshow(predicted_data)
    # # axs[2].set_title(titles[2], weight='bold')
    # # # axs[3].imshow(original_data_masked)
    # # axs[3].set_title(titles[3], weight='bold')
    # predicted_data = 255 * predicted_data
    # predicted_data = predicted_data.astype(np.uint8)
    # cv2.imshow("predicted", predicted_data)
    # cv2.imshow("exp", expected_data)
    # cv2.waitKey(0)
    #
    # plt.subplots_adjust(wspace=0.5)
    # for ax in axs:
    #     ax.axis('off')
    # fig.tight_layout()
    #
    # # results_filename = (f"results-"
    # #                     f"F1-{str(f1).replace('.', '_')}-"
    # #                     f"P-{str(precision).replace('.', '_')}-"
    # #                     f"R-{str(recall).replace('.', '_')}-"
    # #                     f"{input_filename}.png")
    # # fig.savefig(f"{results_path}/{results_filename}", dpi=300)
    #
    # if show:
    #     plt.show()
    #
    # plt.close(fig)

if __name__ == '__main__':
    img = np.random.randint(2400)
    check_model_const('resnet_final.pth', f'dataSet/images/image{img}.jpg', f'dataSet/masks/image{img}.txt')