DATA_CSV = '/home/chris/Documents/SDCND/CarND-Behavioral-Cloning-P3/track-1/driving_log.csv'

lines = []
with open(DATA_CSV) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print(len(lines))
        
images = []
measurements = []
for line in lines:
    image = cv2.imread(line[0])
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)