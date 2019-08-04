import numpy
import numpy.fft.fftpack as fftpack
from PIL import Image
import matplotlib.pyplot as plt

maxWidth = 100

class FFT2:
    def __init__(self):
        self.fig = plt.figure()
        
        originalImage = Image.open('go.jpg')
        (ow,oh) = originalImage.size
        
        if ow>maxWidth :
            monoImageArray = numpy.asarray(originalImage.convert('L').resize((maxWidth,oh*maxWidth//ow)))
        else:
            monoImageArray = numpy.asarray(originalImage.convert('L'))
        
        (self.imageHeight, self.imageWidth) = monoImageArray.shape
        self.samples = numpy.zeros((self.imageHeight, self.imageWidth), dtype=complex)
        self.samplePoints = numpy.zeros((self.imageHeight, self.imageWidth, 4))
        self.fftImage = fftpack.fft2(monoImageArray)
        self.fftImageForPlot = numpy.roll(numpy.roll(numpy.real(self.fftImage), self.imageHeight//2, axis=0), self.imageWidth//2, axis=1)
        self.fftMean = numpy.mean(self.fftImageForPlot)
        self.fftStd = numpy.std(self.fftImageForPlot)
        
        self.axes1 = plt.subplot(2,3,1)
        plt.imshow(monoImageArray, cmap='gray')
        
        self.axes2 = plt.subplot(2,3,2)
        p = plt.imshow(self.fftImageForPlot, cmap='gray')
        p.set_clim(self.fftMean-self.fftStd, self.fftMean+self.fftStd)
        
        self.axes3 = plt.subplot(2,3,4)
        self.axes3.set_aspect('equal')
        self.axes3.set_xlim(0,self.imageWidth)
        self.axes3.set_ylim(self.imageHeight,0)

        self.axes4 = plt.subplot(2,3,5)
        self.axes4.set_aspect('equal')
        self.axes4.set_xlim(0,self.imageWidth)
        self.axes4.set_ylim(self.imageHeight,0)
        
        self.axes5 = plt.subplot(2,3,6)
        self.axes5.set_aspect('equal')
        self.axes5.set_xlim(0,self.imageWidth)
        self.axes5.set_ylim(self.imageHeight,0)
        
        self.bMousePressed = False
        self.mouseButton = 0
        self.bCtrlPressed = False
        self.fig.canvas.mpl_connect('motion_notify_event', self.onMove)
        self.fig.canvas.mpl_connect('button_press_event', self.onButtonPress)
        self.fig.canvas.mpl_connect('button_release_event', self.onButtonRelease)
        self.fig.canvas.mpl_connect('key_press_event', self.onKeyPress)
        self.fig.canvas.mpl_connect('key_release_event', self.onKeyRelease)
        
        plt.show()
    
    def onButtonPress(self, event):
        self.bMousePressed = True
        self.mouseButton = event.button
        self.update(event)
    
    def onButtonRelease(self, event):
        self.bMousePressed = False
        self.mouseButton = 0
    
    def onMove(self, event):
        if self.bMousePressed:
            self.update(event)
    
    def onKeyPress(self, event):
        if event.key == 'control':
            self.bCtrlPressed = True
    
    def onKeyRelease(self, event):
        if event.key == 'control':
            self.bCtrlPressed = False
    
    def update(self, event):
        if event.inaxes != self.axes4:
            return
        
        if event.xdata != None:
            x = (int(event.xdata)+self.imageWidth//2)%self.imageWidth
            y = (int(event.ydata)+self.imageHeight//2)%self.imageHeight
            
            plt.sca(self.axes5)
            plt.cla()
            waveImg = numpy.zeros((self.imageHeight,self.imageWidth))
            waveImg[y,x] = 1
            plt.imshow(numpy.real(fftpack.ifft2(waveImg)), cmap='gray')
            
            if not self.bCtrlPressed:
                bNeedUpdate = False
                if self.samples[y,x] != self.fftImage[y,x] and self.mouseButton == 1: #left button
                   bNeedUpdate = True
                   self.samples[y,x] = self.fftImage[y,x]
                   self.samplePoints[(y-self.imageHeight//2)%self.imageHeight,(x-self.imageWidth//2)%self.imageWidth,0] = 1
                   self.samplePoints[(y-self.imageHeight//2)%self.imageHeight,(x-self.imageWidth//2)%self.imageWidth,3] = 1
                elif self.samples[y,x] != numpy.complex(0.0,0.0) and self.mouseButton == 3: #right button
                   bNeedUpdate = True
                   self.samples[y,x] = numpy.complex(0.0,0.0)
                   self.samplePoints[(y-self.imageHeight//2)%self.imageHeight,(x-self.imageWidth//2)%self.imageWidth,0] = 0
                   self.samplePoints[(y-self.imageHeight//2)%self.imageHeight,(x-self.imageWidth//2)%self.imageWidth,3] = 0
                
                if bNeedUpdate:
                    plt.sca(self.axes4)
                    plt.cla()
                    p = plt.imshow(self.fftImageForPlot, cmap='gray')
                    p.set_clim(self.fftMean-self.fftStd,self.fftMean+self.fftStd)
                    plt.imshow(self.samplePoints)
                    
                    plt.sca(self.axes3)
                    plt.cla()
                    plt.imshow(numpy.real(fftpack.ifft2(self.samples)), cmap='gray')
                    
            else:
                for xi in range(x-self.imageWidth//32, x+self.imageWidth//32):
                    for yi in range(y-self.imageWidth//32, y+self.imageWidth//32):
                        if xi>=self.imageWidth:
                            xx = xi-self.imageWidth
                        else:
                            xx = xi
                        if yi>=self.imageHeight:
                            yy = yi-self.imageHeight
                        else:
                            yy = yi
                        if self.mouseButton == 1: #left button
                            self.samples[yy,xx] = self.fftImage[yy,xx]
                            self.samplePoints[(yy-self.imageHeight//2)%self.imageHeight,(xx-self.imageWidth//2)%self.imageWidth,0] = 1
                            self.samplePoints[(yy-self.imageHeight//2)%self.imageHeight,(xx-self.imageWidth//2)%self.imageWidth,3] = 0.7
                        elif self.mouseButton == 3: #right button
                            self.samples[yy,xx] = numpy.complex(0.0,0.0)
                            self.samplePoints[(yy-self.imageHeight//2)%self.imageHeight,(xx-self.imageWidth//2)%self.imageWidth,0] = 0
                            self.samplePoints[(yy-self.imageHeight//2)%self.imageHeight,(xx-self.imageWidth//2)%self.imageWidth,3] = 0
                
                plt.sca(self.axes4)
                plt.cla()
                plt.imshow(self.samplePoints)
                
                plt.sca(self.axes3)
                plt.cla()
                plt.imshow(numpy.real(fftpack.ifft2(self.samples)), cmap='gray')
            
            self.fig.canvas.draw()

if __name__ == '__main__':
    FFT2()