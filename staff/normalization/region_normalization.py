import numpy as np
import cv2

__all__ = [
    'normalization_five_peaks'
]

def get_peaks(hist, no_peaks, margin):    
    sorted_rows = np.argsort(-hist)
    
    found_peaks = []
    
    for i in range(sorted_rows.shape[0]):
        position = sorted_rows[i]
        
        conditions = np.array([abs(found_peak - position) > margin for found_peak in found_peaks])
        
        if conditions.all():
            found_peaks.append(position)
        
        
        if len(found_peaks) == no_peaks:
            break
    
    
    found_peaks.sort()
    
    return found_peaks

"""
Receives a piece of image and normalizes it to locate the staff
at the center of the image + x1.25 margin at each direction
"""
def normalization_five_peaks(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Histogram            
    wHist = np.zeros(image.shape[0])
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            wHist[i] += (255. - gray_image[i][j])
    
    
    '''
    bar_width = 1
    positions = np.arange(wHist.shape[0])
    plt.bar(positions, wHist, bar_width)
    plt.xticks(positions + bar_width / 2, ('0', '1', '2', '3'))
    plt.show()
    '''
    
    peaks = get_peaks(wHist, 5, 20)
    
    staff_height = peaks[-1] - peaks[0]
    span = staff_height // 2
    
    top = peaks[0] - span
    bottom = peaks[-1] + span
    
    cropped_region = image[max(top,0):min(bottom,image.shape[0])]
    
    pad_top = abs(top) if top < 0 else 0
    pad_bottom = bottom - image.shape[0] if bottom > image.shape[0] else 0

    if pad_top > 0 or pad_bottom > 0:
        cropped_region = np.stack( [np.pad(cropped_region[:,:,c],
                                       [(pad_top, pad_bottom), (0, 0)],
                                       mode='symmetric')
                                 for c in range(3)], axis=2)
    
    return cropped_region
    
    