def classification_label2_image(testing_result_class1,filename):
    import numpy as np
    import cv2
    import scipy.io as sio
    import matplotlib.pyplot as plt
    r=[];g=[];b=[];
    prow=testing_result_class1.shape[0]
    pcolm=testing_result_class1.shape[1]
    im3=np.zeros([prow,pcolm,3],np.uint8)
    im2=np.zeros([prow,pcolm,3],np.uint8)
    for row in range (1,prow):
        for colm in range(1,pcolm):
            i=testing_result_class1[row,colm];
            #black value
            if i==0:
                r=0;
                g=0;
                b=0;
                # dark cyan
            elif i==6:
                r=0;
                g=175;
                b=175;
                # green value
            elif i==3:
                r=0;
                g=255;
                b=0;
                #blue value
            elif i==1:
                r=0;
                g=0;
                b=255;

            #yellow value
            elif i==4:
                r=255;
                g=255;
                b=0;
           #magenta value
            elif i==5:
                r=255;
                g=0;
                b=255;
            #Cyan value
            elif i==2:
                r=0;
                g=255;
                b=255;
            #dark green
            elif i==7:
                r = 0;
                g = 128;
                b = 0;
            #tea color class 8
            elif i==8:
                 r=208;
                 g=240;
                 b=170;
            #Navy Blue
            elif i==9:
                 r=0;
                 g=0;
                 b=128;
            #Aqua 
            elif i==10:
                 r=0;
                 g=128;
                 b=128;
            #medium gray
            elif i==11:
                 r=128;
                 g=128;
                 b=128;				 
            # RGB
            im3[row,colm,0]=np.uint8(r);
            im3[row,colm,1]=np.uint8(g);
            im3[row,colm,2]=np.uint8(b);
            # BGR
            im2[row,colm,0]=np.uint8(b);
            im2[row,colm,1]=np.uint8(g);
            im2[row,colm,2]=np.uint8(r);


    # plt.imshow(im3)
    # plt.axis('off')
    cv2.imwrite(filename+'_cv2.png',im2)
    plt.imsave(filename+"_plt.png",im3)
    #im3.save(filename+"_plt.tiff")
    from PIL import Image
    #data = np.random.randint(0, 255, (10, 10)).astype(np.uint8)
    im = Image.fromarray(im3)
    im.save(filename+'.tif')
    #plt.show()

    return im3


def overlayonorginal(clsfied,filname):
    import cv2
    orgimg = cv2.imread("./Landsat8_for_Classification/10Oct13/testing_image.png") # loading orginal image
    clsfied = cv2.cvtColor(clsfied,cv2.COLOR_RGB2BGR)
    a=0.88
    b=1-a
    if(orgimg.shape != clsfied.shape):
        orgimg = cv2.resize(orgimg,clsfied.shape[1::-1])  #reshaping to match sized of both the images
    dst = cv2.addWeighted(orgimg, a, clsfied, b, 0.0) # overlaying classified image on orginal image
    cv2.imwrite(filname+'_orginal.png',orgimg)
    return dst
