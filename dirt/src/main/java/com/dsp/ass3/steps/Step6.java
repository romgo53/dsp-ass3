package com.dsp.ass3.steps;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

/**
 * Step6 – DIRT Similarity Computation
 *
 * Input:
 * pathA \t pathB \t side \t vector
 *
 * Output:
 * pathA \t pathB \t similarity
 */
public class Step6 {

    /*
     * ============================
     * Mapper
     * ============================
     */
    public static class MapperClass
            extends Mapper<LongWritable, Text, Text, Text> {

        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        public void map(LongWritable key, Text value, Context ctx)
                throws IOException, InterruptedException {

            // Expect: pathA \t pathB \t side \t vector
            String[] parts = value.toString().split("\t", 4);
            if (parts.length != 4)
                return;

            String pathA = parts[0];
            String pathB = parts[1];
            String side = parts[2];
            String vector = parts[3];

            // Key by pair
            outKey.set(pathA + "\t" + pathB);
            outVal.set(side + "\t" + vector);
            ctx.write(outKey, outVal);
        }
    }

    /*
     * ============================
     * Reducer
     * ============================
     */
    public static class ReducerClass
            extends Reducer<Text, Text, Text, DoubleWritable> {

        private final DoubleWritable outVal = new DoubleWritable();

        public enum C {
            KEYS, MISSING_VEC1, MISSING_VEC2, SIM_EMITTED, SIM_ZERO
        }

        @Override
        public void reduce(Text key, Iterable<Text> values, Context ctx)
                throws IOException, InterruptedException {

            String vec1 = null;
            String vec2 = null;
            ctx.getCounter(C.KEYS).increment(1);

            // Collect both vectors
            for (Text v : values) {
                String[] p = v.toString().split("\t", 2);
                if (p.length != 2)
                    continue;

                if (p[0].equals("1"))
                    vec1 = p[1];
                else if (p[0].equals("2"))
                    vec2 = p[1];
            }

            if (vec1 == null) {
                ctx.getCounter(C.MISSING_VEC1).increment(1);
                return;
            }
            if (vec2 == null) {
                ctx.getCounter(C.MISSING_VEC2).increment(1);
                return;
            }

            // Build slot-separated maps
            Map<String, Double> p1X = new HashMap<>();
            Map<String, Double> p1Y = new HashMap<>();
            Map<String, Double> p2X = new HashMap<>();
            Map<String, Double> p2Y = new HashMap<>();

            parseVector(vec1, p1X, p1Y);
            parseVector(vec2, p2X, p2Y);

            double sim = dirtSimilarity(p1X, p1Y, p2X, p2Y);
            if (sim > 0.0) {
                ctx.getCounter(C.SIM_EMITTED).increment(1);
                outVal.set(sim);
                ctx.write(key, outVal);
            } else {
                ctx.getCounter(C.SIM_ZERO).increment(1);
            }
        }

        /*
         * ============================
         * Helpers
         * ============================
         */

        private void parseVector(String vector,
                Map<String, Double> X,
                Map<String, Double> Y) {

            String[] feats = vector.split("!");
            for (String f : feats) {
                // word,slot,mi
                String[] p = f.split(",", 3);
                if (p.length != 3)
                    continue;

                String word = p[0];
                String slot = p[1];
                double mi;

                try {
                    mi = Double.parseDouble(p[2]);
                } catch (NumberFormatException e) {
                    continue;
                }

                if (slot.equals("X"))
                    X.put(word, mi);
                else if (slot.equals("Y"))
                    Y.put(word, mi);
            }
        }

        private double dirtSimilarity(
                Map<String, Double> p1X, Map<String, Double> p1Y,
                Map<String, Double> p2X, Map<String, Double> p2Y) {

            double sumP1x = sum(p1X);
            double sumP1y = sum(p1Y);
            double sumP2x = sum(p2X);
            double sumP2y = sum(p2Y);

            if (sumP1x == 0 || sumP1y == 0 || sumP2x == 0 || sumP2y == 0)
                return 0.0;

            double sharedX = sharedSum(p1X, p2X);
            double sharedY = sharedSum(p1Y, p2Y);

            if (sharedX == 0 || sharedY == 0)
                return 0.0;

            return Math.sqrt(
                    (sharedX / (sumP1x + sumP2x)) *
                            (sharedY / (sumP1y + sumP2y)));
        }

        private double sum(Map<String, Double> m) {
            double s = 0.0;
            for (double v : m.values())
                s += v;
            return s;
        }

        private double sharedSum(Map<String, Double> a, Map<String, Double> b) {
            double s = 0.0;
            for (Map.Entry<String, Double> e : a.entrySet()) {
                Double v = b.get(e.getKey());
                if (v != null) {
                    s += e.getValue() + v;
                }
            }
            return s;
        }
    }
}
