package com.dsp.ass3.steps;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

/**
 * Step4: Build path feature vectors
 *
 * Input:
 * path \t slot \t word \t MI
 *
 * Output:
 * path \t word,slot,mi!word,slot,mi!...
 */
public class Step4 {
    public enum C {
        MAP_LINES_IN,
        MAP_BAD_LINES,
        MAP_EMITTED,

        RED_PATHS,
        RED_FEATURES_TOTAL,
        RED_FEATURES_MAX,
        RED_FEATURES_GT_1000,
        RED_FEATURES_GT_5000
    }

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

            String[] p = value.toString().split("\t");
            ctx.getCounter(C.MAP_LINES_IN).increment(1);
            if (p.length < 4) {
                ctx.getCounter(C.MAP_BAD_LINES).increment(1);
                return;
            }
            ctx.getCounter(C.MAP_EMITTED).increment(1);

            String path = p[0].trim();
            String slot = p[1].trim();
            String word = p[2].trim();
            String mi = p[3].trim();

            outKey.set(path);
            outVal.set(word + "," + slot + "," + mi);
            ctx.write(outKey, outVal);
        }
    }

    /*
     * ============================
     * Reducer
     * ============================
     */
    public static class ReducerClass
            extends Reducer<Text, Text, Text, Text> {

        private final Text outVal = new Text();

        @Override
        public void reduce(Text key, Iterable<Text> values, Context ctx)
                throws IOException, InterruptedException {
            ctx.getCounter(C.RED_PATHS).increment(1);
            long feats = 0;

            StringBuilder sb = new StringBuilder();

            for (Text v : values) {
                feats++;
                sb.append(v.toString()).append("!");
            }
            ctx.getCounter(C.RED_FEATURES_TOTAL).increment(feats);
            if (feats > 1000)
                ctx.getCounter(C.RED_FEATURES_GT_1000).increment(1);
            if (feats > 5000)
                ctx.getCounter(C.RED_FEATURES_GT_5000).increment(1);

            if (sb.length() == 0)
                return;

            // remove trailing "!"
            sb.setLength(sb.length() - 1);
            ctx.getCounter(C.RED_FEATURES_MAX).increment(Math.max(0, feats));

            outVal.set(sb.toString());
            ctx.write(key, outVal);
        }
    }
}
