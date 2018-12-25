import glob
import json
import os
import shutil
import sys

def mAP():
  is_recall = False

  MINOVERLAP = 0.9 # default value (defined in the PASCAL VOC2012 challenge)
  ground_truth_files_dir = 'data/mAP/ground_truth/'
  predicted_files_dir = 'data/mAP/predicted/'
  specific_iou_flagged = False
  results_files_path = "data/mAP/results"
  if os.path.exists(results_files_path):  # if it exist already
      # reset the results directory
      shutil.rmtree(results_files_path)
  os.makedirs(results_files_path)
  """
   throw error and exit
  """
  def error(msg):
    print(msg)
    sys.exit(0)

  """
   Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
  """
  def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
      (goes from the end to the beginning)
      matlab:  for i=numel(mpre)-1:-1:1
                  mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #   range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #   range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
      mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
      matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
      if mrec[i] != mrec[i-1]:
        i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
      (numerical integration)
      matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
      ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

  """
   Convert the lines of a file to a list
  """
  def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
      content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

  """
   Create a "tmp_files/" and "results/" directory
  """
  tmp_files_path = "data/mAP/tmp_files"
  if not os.path.exists(tmp_files_path): # if it doesn't exist already
    os.makedirs(tmp_files_path)

  """
   Ground-Truth
     Load each of the ground-truth files into a temporary ".json" file.
     Create a list of all the class names present in the ground-truth (gt_classes).
  """
  # get a list with the ground-truth files
  ground_truth_files_list = glob.glob(ground_truth_files_dir+'*.txt')
  if len(ground_truth_files_list) == 0:
    error("Error: No ground-truth files found!")
  ground_truth_files_list.sort()
  # dictionary with counter per class
  gt_counter_per_class = {}

  for txt_file in ground_truth_files_list:
    #print(txt_file)
    file_id = txt_file.split(".txt",1)[0]
    file_id = os.path.basename(os.path.normpath(file_id))
    # check if there is a correspondent predicted objects file
    if not os.path.exists(predicted_files_dir + file_id + ".txt"):
      error_msg = "Error. File not found: predicted/" +  file_id + ".txt\n"
      error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
      error(error_msg)
    lines_list = file_lines_to_list(txt_file)
    # create ground-truth dictionary
    bounding_boxes = []
    is_difficult = False
    for line in lines_list:
      try:
        if "difficult" in line:
            class_name, left, top, right, bottom, _difficult = line.split()
            is_difficult = True
        else:
            class_name, left, top, right, bottom = line.split()
      except ValueError:
        error_msg = "Error: File " + txt_file + " in the wrong format.\n"
        error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
        error_msg += " Received: " + line
        error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
        error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
        error(error_msg)

      bbox = left + " " + top + " " + right + " " + bottom
      if is_difficult:
          bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
          is_difficult = False
      else:
          bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
          # count that object
          if class_name in gt_counter_per_class:
            gt_counter_per_class[class_name] += 1
          else:
            # if class didn't exist yet
            gt_counter_per_class[class_name] = 1
    with open(tmp_files_path + "/" + file_id + "_ground_truth.json", 'w') as outfile:
      json.dump(bounding_boxes, outfile)

  gt_classes = list(gt_counter_per_class.keys())
  # let's sort the classes alphabetically
  gt_classes = sorted(gt_classes)
  n_classes = len(gt_classes)
  print(gt_counter_per_class)
  """
   Predicted
     Load each of the predicted files into a temporary ".json" file.
  """
  # get a list with the predicted files
  predicted_files_list = glob.glob(predicted_files_dir + '*.txt')
  predicted_files_list.sort()

  for class_index, class_name in enumerate(gt_classes):
    bounding_boxes = []
    for txt_file in predicted_files_list:
      # the first time it checks if all the corresponding ground-truth files exist
      file_id = txt_file.split(".txt",1)[0]
      file_id = os.path.basename(os.path.normpath(file_id))
      if class_index == 0:
        if not os.path.exists(ground_truth_files_dir + file_id + ".txt"):
          error_msg = "Error. File not found: ground-truth/" +  file_id + ".txt\n"
          error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
          error(error_msg)
      lines = file_lines_to_list(txt_file)
      for line in lines:
        try:
          tmp_class_name, confidence, left, top, right, bottom = line.split()
        except ValueError:
          error_msg = "Error: File " + txt_file + " in the wrong format.\n"
          error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
          error_msg += " Received: " + line
          error(error_msg)
        if tmp_class_name == class_name:
          bbox = left + " " + top + " " + right + " " +bottom
          bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
    # sort predictions by decreasing confidence
    bounding_boxes.sort(key=lambda x:x['confidence'], reverse=True)
    with open(tmp_files_path + "/" + class_name + "_predictions.json", 'w') as outfile:
        json.dump(bounding_boxes, outfile)

  """
   Calculate the AP for each class
  """
  sum_AP = 0.0
  ap_dictionary = {}
  AP_list = []
  # open file to store the results
  with open(results_files_path + "/results.txt", 'w') as results_file:
    results_file.write("# AP and precision/recall per class\n")
    count_true_positives = {}
    for class_index, class_name in enumerate(gt_classes):
      count_true_positives[class_name] = 0
      """
       Load predictions of that class
      """
      predictions_file = tmp_files_path + "/" + class_name + "_predictions.json"
      predictions_data = json.load(open(predictions_file))

      """
       Assign predictions to ground truth objects
      """
      nd = len(predictions_data)
      tp = [0] * nd # creates an array of zeros of size nd
      fp = [0] * nd
      for idx, prediction in enumerate(predictions_data):
        file_id = prediction["file_id"]
        # assign prediction to ground truth object if any
        #   open ground-truth with that file_id
        gt_file = tmp_files_path + "/" + file_id + "_ground_truth.json"
        ground_truth_data = json.load(open(gt_file))
        ovmax = -1
        gt_match = -1
        # load prediction bounding-box
        bb = [ float(x) for x in prediction["bbox"].split() ]
        for obj in ground_truth_data:
          # look for a class_name match
          if obj["class_name"] == class_name:
            bbgt = [ float(x) for x in obj["bbox"].split() ]
            bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
            iw = bi[2] - bi[0] + 1
            ih = bi[3] - bi[1] + 1
            if iw > 0 and ih > 0:
              # compute overlap (IoU) = area of intersection / area of union
              if is_recall:
                ua = (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1)
              else:
                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                        + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
              ov = iw * ih / ua
              if ov > ovmax:
                ovmax = ov
                gt_match = obj
        # set minimum overlap
        min_overlap = MINOVERLAP
        if ovmax >= min_overlap:
            if not bool(gt_match["used"]):
                # true positive
                tp[idx] = 1
                gt_match["used"] = True
                count_true_positives[class_name] += 1
                # update the ".json" file
                with open(gt_file, 'w') as f:
                    f.write(json.dumps(ground_truth_data))
            else:
                # false positive (multiple detection)
                fp[idx] = 1
        else:
          # false positive
          fp[idx] = 1
      # compute precision/recall
      cumsum = 0
      for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
      cumsum = 0
      for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val
      rec = tp[:]
      for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
      prec = tp[:]
      for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (idx+1) #(fp[idx] + tp[idx])
      print(rec[-1])
      print(prec[-1])
      ap, mrec, mprec = voc_ap(rec, prec)
      sum_AP += ap
      text = class_name + " AP = {0:.2f}%".format(ap*100)
      AP_list.append(ap*100)

      print(text)
      """
       Write to results.txt
      """
      rounded_prec = [ '%.2f' % elem for elem in prec ]
      rounded_rec = [ '%.2f' % elem for elem in rec ]
      results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")
      ap_dictionary[class_name] = ap
    results_file.write("\n# mAP of all classes\n")
    mAP = sum_AP / n_classes
    text = "mAP = {0:.2f}%".format(mAP*100)
    results_file.write(text + "\n")
    print(text)
    AP_list.append(mAP*100)
  return AP_list

if __name__ == "__main__":
  mAP()