import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaLiniarRegression {

    public static void main(String[] args) throws Exception {

        DataSource source = new DataSource("C:\\Program Files\\Weka-3-9\\data\\cpu.arff");
        Instances dataset = source.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);

        System.out.println("LINEAR REGRESSION MODEL");
        System.out.println("---------------------");
        LinearRegression lr = new LinearRegression();
        lr.buildClassifier(dataset);

        Evaluation lreval = new Evaluation(dataset);
        lreval.evaluateModel(lr, dataset);
        System.out.println(lreval.toSummaryString());
    }
}
