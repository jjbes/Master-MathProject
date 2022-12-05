import sys
import numpy as np

if __name__ == '__main__':
    from convertInkmlToImg import parse_inkml, genImgFromLGHypotheses, convert_to_imgs, get_traces_data
    from segmenter import generateHypSeg
    # render one image for every symbol in inkml file using GT segmentation
    traces = parse_inkml(sys.argv[1])
    with open(sys.argv[2], 'r') as lg_f: 
        lg = lg_f.read()
    imgs = genImgFromLGHypotheses(traces, lg, 64)
    
    # Add padding and gaussian blur yourself
    # Plot returned images
    import matplotlib.pyplot as plt
    plt.figure()
    N = imgs.shape[0]
    for i in range(N):
        ax = plt.subplot(int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N))), i+1)
        plt.imshow(imgs[i], origin='upper', alpha = 1)
        ax.set_xticklabels([]);ax.set_yticklabels([])
    plt.show()
    plt.savefig('./test.png')

    plt.figure()
    imgs = np.zeros((len(traces), 64, 64))
    for u, (i, trace) in enumerate(traces.items()):
        
        imgs[u] = convert_to_imgs(get_traces_data(traces, [i]), 64)
    hypothesis = generateHypSeg(len(imgs))
    N = len(hypothesis)
    for i in range(N):
        ax = plt.subplot(int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N))), i+1)
        # imgs[hypothesis[i]].sum()
        plt.imshow((255 - imgs[hypothesis[i]]).sum(0).clip(0,255), origin='upper', alpha = 1)
        ax.set_xticklabels([]);ax.set_yticklabels([])
    plt.show()
    plt.savefig('./test2.png')
    print('end')