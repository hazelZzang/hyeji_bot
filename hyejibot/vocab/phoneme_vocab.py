

class PhonemeVocab:
    def __init__(self):
        self.UNICODE_N, self.TOP_N, self.MID_N = 44032, 588, 28
        self.NULL = [0,0]
        self.single = [('ㄱ',' '), ('ㄱ','ㄱ'), ('ㄱ','ㅅ'), ('ㄴ',' '), ('ㄴ','ㅈ'), ('ㄴ','ㅎ'), ('ㄷ',' '),('ㄷ','ㄷ'),
                       ('ㄹ',' '), ('ㄹ','ㄱ'),('ㄹ','ㅁ'), ('ㄹ','ㅂ'), ('ㄹ','ㅅ'), ('ㄹ','ㅌ'), ('ㄹ','ㅍ'), ('ㄹ','ㅎ'),
                       ('ㅁ',' '), ('ㅂ',' '),('ㅂ','ㅂ'),('ㅂ','ㅅ'), ('ㅅ',' '),('ㅅ','ㅅ'), ('ㅇ',' '), ('ㅈ',' '), ('ㅈ','ㅈ'),
                       ('ㅊ',' '), ('ㅋ',' ') , ('ㅌ',' '), ('ㅍ',' '), ('ㅎ',' ')]

        self.top = [('ㄱ',' '), ('ㄱ','ㄱ'), ('ㄴ',' '), ('ㄷ',' '), ('ㄷ','ㄷ'), ('ㄹ',' '), ('ㅁ',' '), ('ㅂ',' '), ('ㅂ','ㅂ'),
                    ('ㅅ',' '), ('ㅅ','ㅅ'), ('ㅇ',' '), ('ㅈ',' '), ('ㅈ','ㅈ'), ('ㅊ',' '), ('ㅋ',' '), ('ㅌ',' '), ('ㅍ',' '), ('ㅎ',' '), (' ',' ')]
        self.mid = [(' ','ㅏ'), (' ','ㅐ'), (' ','ㅑ'), (' ','ㅒ'), (' ','ㅓ'), (' ','ㅔ'), (' ','ㅕ'), (' ','ㅖ'),
                    ('ㅗ',' '), ('ㅗ','ㅏ'), ('ㅗ','ㅐ'), ('ㅗ','ㅣ'), ('ㅛ',' '), ('ㅜ',' '), ('ㅜ','ㅓ'), ('ㅜ','ㅔ'),
                    ('ㅜ','ㅣ'), ('ㅠ',' '), ('ㅡ',' '), ('ㅡ','ㅣ'),(' ','ㅣ'), (' ',' ')]
        self.bot = [(' ',' '), ('ㄱ',' '), ('ㄱ','ㄱ'), ('ㄱ','ㅅ'), ('ㄴ',' '), ('ㄴ','ㅈ'), ('ㄴ','ㅎ'), ('ㄷ',' '), ('ㄹ',' '), ('ㄹ','ㄱ'),
                    ('ㄹ','ㅁ'), ('ㄹ','ㅂ'), ('ㄹ','ㅅ'), ('ㄹ','ㅌ'), ('ㄹ','ㅍ'), ('ㄹ','ㅎ'), ('ㅁ',' '), ('ㅂ',' '), ('ㅂ','ㅅ'), ('ㅅ',' '),
                    ('ㅅ','ㅅ'), ('ㅇ',' '), ('ㅈ',' '), ('ㅊ',' '), ('ㅋ',' '), ('ㅌ',' '), ('ㅍ',' '), ('ㅎ',' ')]

        self.idx2phone = [' ','ㅗ','ㅛ', 'ㅜ', 'ㅠ', 'ㅡ',
                      'ㅏ', 'ㅑ','ㅐ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ','ㅣ',
                      'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.phone2idx = {p:i for i,p in enumerate(self.idx2phone)}
        self.top = [list((self.phone2idx[t[0]],self.phone2idx[t[1]])) for t in self.top]
        self.mid = [list((self.phone2idx[t[0]],self.phone2idx[t[1]])) for t in self.mid]
        self.bot = [list((self.phone2idx[t[0]],self.phone2idx[t[1]])) for t in self.bot]

        self.n_words = len(self.idx2phone)

    def sent2idx(self, word):
        """
        문장 또는 단어를 분리한다.
        :param word: 단어
        :return: [ㄷ,ㅏ,ㄴ,ㅇ,ㅓ]
        """
        char = list(word)

        # 입력 단어를 분리해서 저장
        split_lists = []

        for c in char:
            split_list = []
            char_code = ord(c) - self.UNICODE_N

            # 자음 또는 모음
            if char_code < 0:
                code = ord(c)-ord('ㄱ')
                if code < 0:
                    split_list += [self.NULL,self.NULL,self.NULL]
                elif code < len(self.single):
                    single = [self.phone2idx[i] for i in self.single[code]]
                    split_list += [single,self.NULL,self.NULL]
                else:
                    split_list += [self.NULL,self.mid[code-len(self.single)],self.NULL]
            # 음절
            else:
                # 초성 분리
                top = int(char_code / self.TOP_N)
                split_list.append(self.top[top])

                # 종성 분리
                mid = int((char_code - (self.TOP_N * top)) / self.MID_N)
                split_list.append(self.mid[mid])

                # 종성 분리
                bot = int((char_code - (self.TOP_N * top) - (self.MID_N * mid)))
                split_list.append(self.bot[bot])
            split_lists.append(split_list)
        return split_lists
