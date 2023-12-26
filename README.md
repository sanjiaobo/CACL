# CACL
Complexity Aware Center Loss for Facial Expression Recognition

## Abstract
Deep metric based center loss has been widely used to enhance inter-class separability and intra-class compactness of network features and achieved promissing results in facial expression recognition (FER) recently.  However, existing center loss does not take the complexity of expression samples into consideration, which deteriorates the representativeness of the generated category centers resulting in suboptimal performace. To solve this problem, we propose a novel Complexity Aware Center Loss (CACL) for FER. Specifically, samples in each batch are firstly divided into two groups i.e., simple samples and complex samples  adaptively according to their recognition difficulty. To ensure the representativeness of the obtained center vector, we only use simple samples to calculate the corresponding center. Then, we maintain a suitable distance between the complex samples and the center vector to reduce the interference of complex samples in the model learning process.  Extensive experiments on two benchmark datasets, i.e., RAF-DB and AffectNet8, demonstrate the effectiveness of our formulation

**Dataset**

Download [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset), put it into the dataset folder.

**Pretrained backbone model**

Download the pretrained ResNet18 from [this](https://github.com/amirhfarzaneh/dacl) github repository, and then put it into the pretrained_model directory. We thank the authors for providing their pretrained ResNet model.
