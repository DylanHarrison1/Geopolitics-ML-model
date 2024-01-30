from model import Model


class Instance():
    def __init__(self, feedback):
        """
        Input:
        Feedback data?- Binary
        Model Details- Int
        
        
        """
        self._feedback = feedback
        self._instance = Model()

    def Run(self, epochs):
        """
        Input:
        Epoch num- Int

        """
        while (True):
            yPred = self._instance.calc("??????x")
            self._instance.train(yPred, "??????y") 







            if self._feedback:
                self.__PrintProgress()

    def __PrintProgress():
        pass
    
