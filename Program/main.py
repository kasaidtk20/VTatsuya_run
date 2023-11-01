import cv2
import mediapipe as mp
import time
import numpy as np
import pyautogui as au
import time

video_path = "../OBS/"

def landmark2np(hand_landmarks):
    li = []
    for j in (hand_landmarks.landmark):
        li.append([j.x, j.y, j.z])

    return np.array(li) - li[0]

def manual_cos(A, B):
    dot = np.sum(A*B, axis=-1)
    A_norm = np.linalg.norm(A, axis=-1)
    B_norm = np.linalg.norm(B, axis=-1)
    cos = dot / (A_norm*B_norm+1e-7)

    return cos[1:].mean()

def hotkey(key):
    au.keyDown("shift")
    au.keyDown(key)
    au.keyUp(key)
    au.keyUp("shift")

def decision(img, count, score, num, videoname, key):
    cv2.putText(img, videoname, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 255), thickness=2)
    print(count)
    if count > 10:
        hotkey(key)
        score[0][num] = 0
        score[1][num] = 0
        count = 0

        #動画再生が終わったらオフ
        try:
            cap = cv2.VideoCapture(video_path + videoname + ".avi")
            playtime = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            time.sleep(playtime)
        except ZeroDivisionError:
            print("ZeroDivisionError")

        hotkey(key)


def main():
    saved_array = [None, None, None, None, None]
    start = -100
    score = [[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]
    saved_no = 0
    count = 0
    bool = True
    posemode = True

    #ハート(右手のみ)
    saved_array[0] =   [[ 0,           0,           0         ], \
                        [ 0.11025721,  0.01008397,  0.03452676], \
                        [ 0.20023713, -0.01660782,  0.036063  ], \
                        [ 0.26233137, -0.00983673,  0.02138756], \
                        [ 0.29732639,  0.03137869,  0.00585059], \
                        [ 0.18576434, -0.20963681,  0.05045359], \
                        [ 0.26224497, -0.25166613,  0.03604897], \
                        [ 0.30577269, -0.21000111,  0.01989812], \
                        [ 0.32530433, -0.15743607,  0.00955754], \
                        [ 0.1656324,  -0.22307616,  0.0119056 ], \
                        [ 0.265531,   -0.27099502, -0.0067362 ], \
                        [ 0.31339389, -0.21905226, -0.02148977], \
                        [ 0.32299256, -0.1531691,  -0.02667798], \
                        [ 0.14844087, -0.21542561, -0.02788846], \
                        [ 0.24670646, -0.26684111, -0.04200024], \
                        [ 0.29967216, -0.21398932, -0.04584462], \
                        [ 0.31795806, -0.15160489, -0.04443638], \
                        [ 0.13839006, -0.1842528,  -0.06713445], \
                        [ 0.22235087, -0.21683979, -0.07725446], \
                        [ 0.27446041, -0.19547015, -0.07724424], \
                        [ 0.30394027, -0.16142541, -0.07433322]]
    
    #OK(右手)
    saved_array[1] =   [[ 0,           0,           0         ] , \
                        [ 0.0805029,  -0.04128224, -0.02446689] , \
                        [ 0.1578871,  -0.10647166, -0.04686035] , \
                        [ 0.22373086, -0.14705533, -0.07257053] , \
                        [ 0.26758283, -0.19353688, -0.0950715 ] , \
                        [ 0.11851576, -0.27688688, -0.01972196] , \
                        [ 0.19619009, -0.30794787, -0.07244794] , \
                        [ 0.23979068, -0.26512319, -0.11340233] , \
                        [ 0.26410395, -0.21784455, -0.13319196] , \
                        [ 0.08225876, -0.31701997, -0.03304238] , \
                        [ 0.12553394, -0.43915871, -0.06084637] , \
                        [ 0.16152698, -0.51868424, -0.07909413] , \
                        [ 0.18554366, -0.59006032, -0.08923314] , \
                        [ 0.03837371, -0.31751132, -0.05426572] , \
                        [ 0.04159129, -0.44926071, -0.08476676] , \
                        [ 0.04697639, -0.5352833,  -0.09786206] , \
                        [ 0.05386436, -0.60920751, -0.10210402] , \
                        [-0.00847921, -0.28231382, -0.07870905] , \
                        [-0.04389423, -0.37197074, -0.10559092] , \
                        [-0.05890991, -0.43707216, -0.10935943] , \
                        [-0.06374988, -0.49988201, -0.10662492]]
    
    #ピース(右手)
    saved_array[2] =   [[ 0,           0,           0         ] , \
                        [ 0.04728431, -0.04837364, -0.03962957] , \
                        [ 0.06097072, -0.12322956, -0.0607649 ] , \
                        [ 0.0146319,  -0.16611773, -0.08119416] , \
                        [-0.03501377, -0.19780743, -0.09858779] , \
                        [ 0.0294866,  -0.25869048, -0.03119527] , \
                        [ 0.04480198, -0.35631108, -0.06317865] , \
                        [ 0.0553405,  -0.41773039, -0.0830047 ] , \
                        [ 0.06155953, -0.47222596, -0.09476836] , \
                        [-0.02147689, -0.24970669, -0.03474671] , \
                        [-0.05594991, -0.3588776,  -0.07406022] , \
                        [-0.07840167, -0.43008059, -0.09758282] , \
                        [-0.0998901,  -0.49459127, -0.10790719] , \
                        [-0.0576295,  -0.20701247, -0.0424214 ] , \
                        [-0.07299121, -0.24624556, -0.09859692] , \
                        [-0.03545707, -0.17956108, -0.1068169 ] , \
                        [-0.01108202, -0.13626683, -0.09552935] , \
                        [-0.08432534, -0.14617664, -0.05244652] , \
                        [-0.0871689,  -0.16777563, -0.09652515] , \
                        [-0.05485141, -0.12619287, -0.09690648] , \
                        [-0.03030489, -0.09880334, -0.08570588]]
    #ピース(左手)
    saved_array[3] =   [[ 0,           0,           0         ] , \
                        [-0.04071969, -0.04245937, -0.02748401] , \
                        [-0.05564672, -0.11146504, -0.04600084] , \
                        [-0.02246982, -0.15717232, -0.06628745] , \
                        [ 0.01286888, -0.19249398, -0.08489263] , \
                        [-0.03278506, -0.23334986, -0.02769995] , \
                        [-0.04525495, -0.3230418,  -0.05345303] , \
                        [-0.05288643, -0.38047734, -0.06843718] , \
                        [-0.05689913, -0.43030137, -0.07722365] , \
                        [ 0.01114357, -0.2272625,  -0.03487923] , \
                        [ 0.03657234, -0.33509469, -0.06436635] , \
                        [ 0.05386949, -0.40770444, -0.08113762] , \
                        [ 0.07167172, -0.467444,   -0.0878354 ] , \
                        [ 0.04565346, -0.19281238, -0.04538521] , \
                        [ 0.04943579, -0.23347908, -0.08582776] , \
                        [ 0.02314967, -0.17152113, -0.08836962] , \
                        [ 0.00925666, -0.12772894, -0.07726106] , \
                        [ 0.07182235, -0.14411724, -0.05792571] , \
                        [ 0.07012814, -0.18296146, -0.08559169] , \
                        [ 0.04386741, -0.14433557, -0.07928747] , \
                        [ 0.02591008, -0.11158884, -0.06686648]]

    cap = cv2.VideoCapture(1)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2)
    mp_draw = mp.solutions.drawing_utils
    

    """
    while True:
        print("in true")
    """

    while posemode:
        _, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            cnt = len(results.multi_hand_landmarks)  #手の数(max:2)
            for idx in range(cnt):
                try:
                    hand_landmarks = results.multi_hand_landmarks[idx]

                    for i, lm in enumerate(hand_landmarks.landmark):
                        height, width, channel = img.shape
                        cx, cy = int(lm.x * width), int(lm.y * height)
                        cv2.putText(img, str(i+1), (cx+10, cy+10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.circle(img, (cx, cy), 4, (255, 0, 255), cv2.FILLED)
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    """
                    #ポーズ登録
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        saved_array[0] = landmark2np(hand_landmarks)
                        start = time.time()
                        saved_no = 0
                        print(saved_array[0])
                    """
                    
                    # cos類似度でチェック
                    for n in range(4):
                        if saved_array[n] is not None:
                            now_array = landmark2np(hand_landmarks)
                            score[idx][n] = manual_cos(saved_array[n], now_array)

                except IndexError:
                    print("IndexError")

        #判定
        if time.time() - start < 3:
            cv2.putText(img, f'No.{saved_no} saved', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), thickness=2)

        elif score[0][0] > 0.96 or score[1][0] > 0.96:
            count += 1
            if bool == True:
                decision(img, count, score, 0, "heart0", "a")
                bool = not bool
    
            else:
                decision(img, count, score, 0, "heart1", "s")
                bool = not bool

        elif score[0][1] > 0.985 or score[1][1] > 0.985:
            count += 1
            decision(img, count, score, 1, "ok", "d")

        elif (score[0][2] > 0.95 and score[1][3] > 0.95) or (score[1][2] > 0.95 and score[0][3] > 0.95):
            count += 1
            if bool == True:
                decision(img, count, score, 2, "tulip0", "f")
                bool = not bool
    
            else:
                decision(img, count, score, 2, "tulip1", "g")
                bool = not bool

        else:
            count = 0

        #print([row[2] for row in score], "\n", [row[3] for row in score])

        #表示
        cv2.imshow("DetectMonitor", img)

        """
        #"DetectMonitor"をアクティブウィンドウ化
        wndw = gw.getWindowsWithTitle("DetectMonitor")[0]
        wndw.activate()
    
        if cv2.waitKey(1) & 0xFF == ord('p'):
            posemode = not posemode
        """
            
        if cv2.waitKey(1) & 0xFF == ord('Q'): break

    """
    while not posemode:
        wndw.activate()
        score = [[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]
        if cv2.waitKey(1) & 0xFF == ord('p'):
            posemode = not posemode
    """


"""
if __name__ == "__main__":
    main()
"""