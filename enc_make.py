import base64
from Crypto import Random
from Crypto.Cipher import AES
import hashlib
import yaml

# 블럭사이즈에 대한 패딩로직
BS = 16
pad = lambda s: s + (BS - len(s.encode('utf-8')) % BS) * chr(BS - len(s.encode('utf-8')) % BS)
unpad = lambda s: s[:-ord(s[len(s)-1:])]

class AESCipher:
    def __init__(self):
        self.key = hashlib.sha256().digest()

    def encrypt(self, raw):
        raw = pad(raw)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(raw.encode('utf-8')))

    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        iv = enc[:16]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(enc[16:]))


with open('conf/db_conf.yaml') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)

tibero = conf.get('tibero', {})
user = tibero.get('user')
password = tibero.get('password')
url = tibero.get('url')

# 데이터 암호화
encrypted_user = AESCipher().encrypt(user)
encrypted_password = AESCipher().encrypt(password)
encrypted_url = AESCipher().encrypt(url)

enc_tibero = {'tibero': {'class_name': 'com.tmax.tibero.jdbc.TbDriver',
                         'url': encrypted_url,
                         'user': encrypted_user,
                         'password': encrypted_password},
              'jdbc_driver': 'tibero6-jdbc.jar',
              'JAVA_HOME': "C:/JAVA",
              'default_path': ""
              }

with open('conf/db_conf_enc_prod.yaml', 'w') as f:
    yaml.dump(enc_tibero, f)
