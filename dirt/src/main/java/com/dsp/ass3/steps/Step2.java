package com.dsp.ass3.steps;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

/**
 * Step2: build the marginal counts needed for MI(p,slot,word)
 *
 * Input (from Step1):
 * path \t slot \t word \t count
 *
 * Output:
 * PSW \t path \t slot \t word \t c(p,s,w)
 * PS* \t path \t slot \t \t c(p,s,*)
 * *SW \t \t slot \t word \t c(*,s,w)
 */
public class Step2 {

    public static class MapperClass extends Mapper<LongWritable, Text, Text, IntWritable> {

        private final IntWritable outVal = new IntWritable();
        private final Text outKey = new Text();

        public enum C {
            LINES, BAD, EMIT
        }

        @Override
        public void map(LongWritable key, Text value, Context ctx)
                throws IOException, InterruptedException {

            // Expect: path \t slot \t word \t count
            String[] parts = value.toString().split("\t");
            ctx.getCounter(C.LINES).increment(1);

            if (parts.length < 4) {
                ctx.getCounter(C.BAD).increment(1);
                return;
            }

            String path = parts[0].trim();
            String slot = parts[1].trim();
            String word = parts[2].trim();

            int count;
            try {
                count = Integer.parseInt(parts[3].trim());
            } catch (NumberFormatException e) {
                return;
            }
            ctx.getCounter(C.EMIT).increment(3);

            outVal.set(count);

            // 1) c(p,s,w)
            outKey.set("PSW\t" + path + "\t" + slot + "\t" + word);
            ctx.write(outKey, outVal);

            // 2) c(p,s,*)
            outKey.set("PS*\t" + path + "\t" + slot);
            ctx.write(outKey, outVal);

            // 3) c(*,s,w)
            outKey.set("*SW\t" + slot + "\t" + word);
            ctx.write(outKey, outVal);
        }
    }

    public static class ReducerClass extends Reducer<Text, IntWritable, Text, IntWritable> {

        private final IntWritable outVal = new IntWritable();

        public enum CR {
            KEYS
        }

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context ctx)
                throws IOException, InterruptedException {
            ctx.getCounter(CR.KEYS).increment(1);

            int sum = 0;
            for (IntWritable v : values)
                sum += v.get();

            outVal.set(sum);
            ctx.write(key, outVal);
        }
    }
}
