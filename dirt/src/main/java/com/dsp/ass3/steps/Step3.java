package com.dsp.ass3.steps;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import com.dsp.ass3.steps.Step2.ReducerClass.CR;

/**
 * Step3: Compute MI(p,slot,word)
 *
 * Input:
 * PSW \t path \t slot \t word \t c_psw
 * PS* \t path \t slot \t \t c_ps
 * *SW \t \t slot \t word \t c_sw
 *
 * Output:
 * path \t slot \t word \t MI
 */
public class Step3 {

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
            if (p.length < 3)
                return;

            ctx.getCounter(CR.KEYS).increment(1);

            String type = p[0];

            if (type.equals("PSW")) {
                // PSW \t path \t slot \t word \t c_psw
                if (p.length != 5)
                    return;

                String path = p[1];
                String slot = p[2];
                String word = p[3];
                String c_psw = p[4];

                // group under slot|word keys (prefix 1)
                outKey.set("1\t" + slot + "\t" + word);
                outVal.set("PSW\t" + path + "\t" + c_psw);
                ctx.write(outKey, outVal);

            } else if (type.equals("*SW")) {
                // *SW \t slot \t word \t c_sw (Step2 outputs 4 cols)
                if (p.length != 4)
                    return;

                String slot = p[1];
                String word = p[2];
                String c_sw = p[3];

                // 0 S = slot totals first
                outKey.set("0\tS\t" + slot);
                outVal.set("S\t" + c_sw);
                ctx.write(outKey, outVal);

                // 1 = slot|word group
                outKey.set("1\t" + slot + "\t" + word);
                outVal.set("*SW\t" + c_sw);
                ctx.write(outKey, outVal);

            } else if (type.equals("PS*")) {
                // PS* \t path \t slot \t c_ps (Step2 outputs 4 cols)
                if (p.length != 4)
                    return;

                String path = p[1];
                String slot = p[2];
                String c_ps = p[3];
                // 0 P = path-slot totals first
                outKey.set("0\tP\t" + path + "\t" + slot);
                outVal.set("PS*\t" + c_ps);
                ctx.write(outKey, outVal);
            }
        }

    }

    /*
     * ============================
     * Reducer
     * ============================
     */
    public static class ReducerClass extends Reducer<Text, Text, Text, DoubleWritable> {

        private final Map<String, Integer> pathSlotTotals = new HashMap<>();
        private final Map<String, Long> slotTotals = new HashMap<>();

        private final DoubleWritable outVal = new DoubleWritable();
        private final Text outKey = new Text();

        public enum C {
            SLOT_TOTAL_KEYS, PATHSLOT_TOTAL_KEYS, SLOTWORD_KEYS, MI_EMITTED, MI_NONPOS, MI_BAD
        }

        @Override
        public void reduce(Text key, Iterable<Text> values, Context ctx)
                throws IOException, InterruptedException {

            String[] k = key.toString().split("\t", -1);
            if (k.length < 2)
                return;

            // ---------- Prefix 0: totals (must be processed before prefix 1) ----------
            if (k[0].equals("0")) {
                ctx.getCounter(C.SLOT_TOTAL_KEYS).increment(1);
                ctx.getCounter(C.PATHSLOT_TOTAL_KEYS).increment(1);
                // 0 S slot => sum c_sw for slot to get c_s
                if (k[1].equals("S") && k.length == 3) {
                    String slot = k[2];
                    long sum = 0;
                    for (Text v : values) {
                        String[] p = v.toString().split("\t");
                        if (p[0].equals("S"))
                            sum += Long.parseLong(p[1]);
                    }
                    slotTotals.put(slot, sum);
                    return;
                }

                // 0 P path slot => store c_ps
                if (k[1].equals("P") && k.length == 4) {
                    String path = k[2];
                    String slot = k[3];
                    int sum = 0;
                    for (Text v : values) {
                        String[] p = v.toString().split("\t");
                        if (p[0].equals("PS*"))
                            sum += Integer.parseInt(p[1]);
                    }
                    pathSlotTotals.put(path + "\t" + slot, sum);
                    return;
                }

                return;
            }

            // ---------- Prefix 1: compute MI for slot|word ----------
            if (!k[0].equals("1") || k.length != 3)
                return;
            ctx.getCounter(C.SLOTWORD_KEYS).increment(1);

            String slot = k[1];
            String word = k[2];

            long c_s = slotTotals.getOrDefault(slot, 0L);
            if (c_s == 0)
                return;

            int c_sw = 0;
            Map<String, Integer> pswCounts = new HashMap<>();

            for (Text v : values) {
                String[] p = v.toString().split("\t");
                if (p[0].equals("*SW")) {
                    c_sw = Integer.parseInt(p[1]);
                } else if (p[0].equals("PSW")) {
                    String path = p[1];
                    int c_psw = Integer.parseInt(p[2]);
                    pswCounts.put(path, c_psw);
                }
            }

            if (c_sw == 0)
                return;

            for (Map.Entry<String, Integer> e : pswCounts.entrySet()) {
                String path = e.getKey();
                int c_psw = e.getValue();

                Integer c_ps = pathSlotTotals.get(path + "\t" + slot);
                if (c_ps == null || c_ps == 0)
                    continue;

                double mi = Math.log((double) c_psw * (double) c_s / ((double) c_ps * (double) c_sw));

                if (!Double.isFinite(mi)) {
                    ctx.getCounter(C.MI_BAD).increment(1);
                    continue;
                }
                if (mi <= 0.0) {
                    ctx.getCounter(C.MI_NONPOS).increment(1);
                    continue;
                }
                ctx.getCounter(C.MI_EMITTED).increment(1);

                outKey.set(path + "\t" + slot + "\t" + word);
                outVal.set(mi);
                ctx.write(outKey, outVal);
            }
        }
    }
}
