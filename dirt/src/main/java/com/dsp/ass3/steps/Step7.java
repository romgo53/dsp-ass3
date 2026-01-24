package com.dsp.ass3.steps;

import java.io.IOException;
import java.util.Locale;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import com.dsp.ass3.utils.Stemmer;

public class Step7 {

    private static String stemToken(String w) {
        w = w.toLowerCase(Locale.ENGLISH).replaceAll("[^a-z]", "");
        if (w.isEmpty())
            return "";
        Stemmer s = new Stemmer();
        s.add(w.toCharArray(), w.length());
        s.stem();
        return s.toString();
    }

    private static String normalizePath(String path) {
        String[] toks = path.trim().split("\\s+");
        if (toks.length == 0)
            return "";

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < toks.length; i++) {
            String t = toks[i];

            if (t.equalsIgnoreCase("X"))
                t = "X";
            else if (t.equalsIgnoreCase("Y"))
                t = "Y";
            else if (i == 1)
                t = stemToken(t); // stem ONLY verb after X
            else
                t = t.toLowerCase(); // lowercase preps

            if (sb.length() > 0)
                sb.append(' ');
            sb.append(t);
        }
        return sb.toString();
    }

    private static void setCanonicalKey(Text outKey, String a, String b) {
        if (a.compareTo(b) <= 0)
            outKey.set(a + "\t" + b);
        else
            outKey.set(b + "\t" + a);
    }

    /** Input A: Step6 similarity lines: pathA \t pathB \t sim */
    public static class SimMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String[] parts = value.toString().split("\t");
            if (parts.length != 3)
                return;

            String a = normalizePath(parts[0].trim());
            String b = normalizePath(parts[1].trim());
            String sim = parts[2].trim();

            setCanonicalKey(outKey, a, b);
            outVal.set("S\t" + sim);
            ctx.write(outKey, outVal);
        }
    }

    /** Input B: Gold labels: pathA \t pathB \t label */
    public static class GoldMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String[] parts = value.toString().split("\t");
            if (parts.length != 3)
                return;

            String a = normalizePath(parts[0].trim());
            String b = normalizePath(parts[1].trim());
            String label = parts[2].trim();

            setCanonicalKey(outKey, a, b);
            outVal.set("G\t" + label);
            ctx.write(outKey, outVal);
        }
    }

    public static class ReducerClass extends Reducer<Text, Text, Text, Text> {
        private final Text outVal = new Text();

        public enum C {
            KEYS, HAS_SIM, HAS_GOLD, EMITTED, DROPPED
        }

        @Override
        public void reduce(Text key, Iterable<Text> values, Context ctx)
                throws IOException, InterruptedException {
            ctx.getCounter(C.KEYS).increment(1);

            String sim = null;
            String label = null;

            for (Text v : values) {
                String[] p = v.toString().split("\t", 2);
                if (p.length != 2)
                    continue;

                if (p[0].equals("S"))
                    sim = p[1];
                else if (p[0].equals("G"))
                    label = p[1];
            }

            if (sim == null || label == null) {
                ctx.getCounter(C.DROPPED).increment(1);
                return;
            }

            ctx.getCounter(C.HAS_SIM).increment(1);
            ctx.getCounter(C.HAS_GOLD).increment(1);

            outVal.set(sim + "\t" + label);
            ctx.write(key, outVal);
            ctx.getCounter(C.EMITTED).increment(1);

        }
    }
}
