import numpy as np
from lib.nms.nms_wrapper import nms
from lib.utils.config import cfg
from lib.text_connect.text_proposal_connector import TextProposalConnector
from lib.text_connect.text_proposal_connector_oriented import TextProposalConnector as TextProposalConnectorOriented


class TextConnector:
    def __init__(self):
        self.mode = cfg["TEXT"]["DETECT_MODE"]
        if self.mode == "H":
            self.text_proposal_connector = TextProposalConnector()
        elif self.mode == "O":
            self.text_proposal_connector = TextProposalConnectorOriented()

    def detect(self, text_proposals, scores, size):
        # 删除得分较低的proposal
        keep_inds = np.where(scores > cfg["TEXT"]["TEXT_PROPOSALS_MIN_SCORE"])[0]
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        # 按得分排序
        sorted_indices = np.argsort(scores.ravel())[::-1]
        text_proposals, scores = text_proposals[sorted_indices], scores[sorted_indices]

        # 对proposal做nms
        # print('text_proposals, scores', text_proposals.shape, scores.shape)

        keep_inds = nms(np.hstack((text_proposals, scores)), cfg["TEXT"]["TEXT_PROPOSALS_NMS_THRESH"])
        # keep_inds = soft_nms(np.hstack((text_proposals, scores)),threshold=TextLineCfg.TEXT_PROPOSALS_NMS_THRESH)
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]
        # 获取检测结果
        text_recs = self.text_proposal_connector.get_text_lines(text_proposals, scores, size)
        keep_inds = self.filter_boxes(text_recs)
        return text_proposals, scores, text_recs[keep_inds]

    def filter_boxes(self, boxes):
        heights = np.zeros((len(boxes), 1), np.float)
        widths = np.zeros((len(boxes), 1), np.float)
        scores = np.zeros((len(boxes), 1), np.float)
        index = 0
        for box in boxes:
            heights[index] = (abs(box[5] - box[1]) + abs(box[7] - box[3])) / 2.0 + 1
            widths[index] = (abs(box[2] - box[0]) + abs(box[6] - box[4])) / 2.0 + 1
            scores[index] = box[8]
            index += 1
        # LINE_MIN_SCORE合成的矩形框得分
        return np.where((widths / heights > cfg["TEXT"]["MIN_RATIO"]) & (scores > cfg["TEXT"]["LINE_MIN_SCORE"]) &
                        (widths > (cfg["ANCHOR_WIDTH"] * cfg["TEXT"]["MIN_NUM_PROPOSALS"])))[0]