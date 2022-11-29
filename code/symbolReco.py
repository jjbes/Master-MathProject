################################################################
# symbolReco.py
#
# Program that select hypothesis of segmentation
#
# Author: H. Mouchere, Dec. 2018
# Copyright (c) 2018, Harold Mouchere
################################################################
import sys
import random
import itertools
import sys, getopt
from convertInkmlToImg import parse_inkml,get_traces_data, getStrokesFromLG, convert_to_imgs, parseLG
from skimage.io import imsave
import torch
from torchvision import transforms
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RECOGNITION_PATH= "./model/recognition.pt"
transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor()])

def usage():
    print ("usage: python3 symbolReco.py [-s] [-o fname][-w weigthFile] inkmlfile lgFile ")
    print ("     inkmlfile  : input inkml file name ")
    print ("     lgFile     : input LG file name")
    print ("     -o fname / --output fname : output file name (LG file)")
    print ("     -w fname / --weight fname : weight file name (nn pytorch file)")
    print ("     -s         : save hyp images")

"""
take an hypothesis (from LG = list of stroke index), select the corresponding strokes (from allTraces) and 
return the probability of being each symbol as a dictionnary {class_0 : score_0 ; ... class_i : score_i } 
Keep only the classes with a score higher than a threshold
"""

def computeClProb(alltraces, hyp, min_threshol, saveIm = False):
    im = convert_to_imgs(get_traces_data(alltraces, hyp[1]), 32)
    if saveIm:
        imsave(hyp[0] + '.png', im)
    # create the list of possible classes (maybe connected to your classifier ???)
    classes = ['!', '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A_', 'B_', 'C_', 'Delta', 'E_', 'F_', 'G_', 'H_', 'I_', 'L_','M_','N_','P_','R_','S_','T_','V_','X_','Y_','[',']','a','alpha','b','beta','c','cos','d','div','div_op','dot','e','exists','f','forall','g','gamma','geq','gt','h','i','in','infty','int','j','k','l','lambda','ldots','leq','lim','log','lt','m','mu','n','neq','o','p','phi','pi','pipe','pm','prime','q','r','rightarrow','s','sigma','sin','sqrt','sum','t','tan','theta','times','u','v','w','x','y','z','{','}']
    
    model = torch.load(RECOGNITION_PATH)
    model.eval()

    im = Image.fromarray(im)
    im = transform(im)
    im = im.to(device)
    im = torch.reshape(im, (1, 1, 32, 32))
    outputs = model(im)
    sm = torch.nn.functional.softmax(outputs, dim=1)
    result={}
    for i, val in enumerate(sm[0]):
      if val.item() > min_threshol:
        result[classes[i]]=val.item()

    return result

def main():

    try:
        opts, args = getopt.getopt(sys.argv[1:], "so:w:", ["output=", "weight="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    if len(args) != 2:
        print("Not enough parameters")
        usage()
        sys.exit(2)
    inputInkml = args[0]
    inputLG = args[1]
    saveimg = False
    outputLG = ""
    weightFile = "myweight.nn"

    for o, a in opts:
        if o in ("-s"):
            saveimg = True
        elif o in ("-o", "--output"):
            outputLG = a
        elif o in ("-w", "--weight"):
            weightFile = a
        else:
            usage()
            assert False, "unhandled option"

    traces = parse_inkml(inputInkml)
    hyplist = open(inputLG, 'r').readlines()
    hyplist = parseLG(hyplist)
    output = ""
    for h in hyplist:
        # for each hypo, call the classifier and keep only slected classes (only the best or more)
        prob_dict = computeClProb(traces, h, 0.2, saveimg)
        #rewrite the new LG
        for cl, prob in prob_dict.items():
            output += "O,"+ h[0]+","+cl+","+str(prob)+","+",".join([str(s) for s in h[1]]) + "\n"
    if outputLG != "":
        with open(outputLG, "w") as text_file:
            print(output, file=text_file)
    else:
        print(output)


if __name__ == "__main__":
    # execute only if run as a script
    main()
