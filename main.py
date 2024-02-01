from model import Model


class Instance():
    def __init__(self, feedback, dbListIn, dbListOut):
        """
        Input:
        feedback data, do we want feedback, Binary - 
        Model Details, Int - 
        dbList, list of data used - 
        
        """
        self._feedback = feedback
        self._instance = Model()

    def Run(self, epochs):
        """
        Input:
        epochs, Int - 

        """
        

        for i in epochs:
            #Open relevant csv files here
            docOpen = True

            while (docOpen):
                yPred = self._instance.calc("??????x")
                self._instance.train(yPred, "??????y") 

                if self._feedback:
                    self.__PrintProgress()

                #When csv files are depleted, close them.
                

        #Final data print goes here
                    

    def __PrintProgress():
        pass
    
