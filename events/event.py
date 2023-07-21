
from abc import ABC

class Listener(ABC):
    def receive(self, event, modifier, data):
        raise NotImplementedError



class Event:
    

    def __init__(self, name='main_event'):
        self.name = name
        self._listeners =  []

    

    
    def attach(self, listener: Listener):
        if issubclass(listener, Listener):
            self._listeners.append(listener)

    
    def detach(self, listener):

        try:
            self._listeners.remove(listener)
        except ValueError:
            pass
    
    def notify(self, event: str, modifier = None, data=None):
        for lnr in  self._listeners:
            if modifier != lnr:
                lnr.receive(event, modifier, data)


_main_subject = Event(name='main_event')