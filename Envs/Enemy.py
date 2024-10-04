

class Enemy:
    def __init__(self,_hp:int,x_pos:int,y_pos:int,_index:int = None) -> None:
        self.hp = _hp
        self.pos = (x_pos,y_pos)
        self.index = _index

    def be_attacked(self,damage:int):
        self.hp -= damage
        if self.hp <= 0:
            return True,self.hp
        else:
            return False,self.hp
    