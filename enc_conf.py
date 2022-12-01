import base64
from Cryptodome import Random
from Cryptodome.Cipher import AES
import hashlib
import yaml

# 블럭사이즈에 대한 패딩로직
BS = 16
pad = lambda s: s + (BS - len(s.encode('utf-8')) % BS) * chr(BS - len(s.encode('utf-8')) % BS)
unpad = lambda s: s[:-ord(s[len(s) - 1:])]


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


def read_db_conf():
    with open('conf/db_conf_enc_adm.yaml') as f:
        conf_enc = yaml.load(f, Loader=yaml.FullLoader)

    tibero = conf_enc.get('tibero', {})
    user = tibero.get('user')
    password = tibero.get('password')
    url = tibero.get('url')
    class_name = tibero.get('class_name')

    # 데이터 복호화
    decrypted_user = AESCipher().decrypt(user).decode('utf-8')
    decrypted_password = AESCipher().decrypt(password).decode('utf-8')
    decrypted_url = AESCipher().decrypt(url).decode('utf-8')

    jdbc_driver = conf_enc.get('jdbc_driver', {})
    JAVA_HOME = conf_enc.get('JAVA_HOME', {})

    return decrypted_user, decrypted_password, decrypted_url, jdbc_driver, JAVA_HOME, class_name
