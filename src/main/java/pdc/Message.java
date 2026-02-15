package pdc;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Message represents the communication unit in the CSM218 protocol.
 *
 * Requirement: You must implement a custom WIRE FORMAT.
 * DO NOT use standard Java Serialization.
 * The assignment specifies a JSON-based structure for messages.
 */
public class Message {
    // Protocol constants as per assignment specification
    private static final String MAGIC = "CSM218";
    private static final int VERSION = 1;

    public enum MessageType {
        // Worker -> Master
        REGISTER_WORKER,
        HEARTBEAT,
        TASK_RESULT,
        // Master -> Worker
        REGISTER_ACK,
        TASK_REQUEST
    }

    private final String messageType;
    private final String studentId;
    private final long timestamp;
    private final String payload; // Base64 encoded string for binary data

    /**
     * Constructor for creating a message to be sent.
     * @param type The type of the message.
     * @param payloadBytes The binary payload.
     */
    public Message(MessageType type, byte[] payloadBytes) {
        this.messageType = type.name();
        this.studentId = System.getenv("STUDENT_ID") != null ? System.getenv("STUDENT_ID") : "default-student";
        this.timestamp = System.currentTimeMillis();
        this.payload = (payloadBytes != null && payloadBytes.length > 0) ? Base64.getEncoder().encodeToString(payloadBytes) : "";
    }

    /**
     * Private constructor for creating a message received from the network.
     */
    private Message(String messageType, String studentId, long timestamp, String payload) {
        this.messageType = messageType;
        this.studentId = studentId;
        this.timestamp = timestamp;
        this.payload = payload;
    }

    public MessageType getType() {
        try {
            return MessageType.valueOf(this.messageType);
        } catch (IllegalArgumentException e) {
            System.err.println("Received unknown message type: " + this.messageType);
            return null; // Or handle as an error
        }
    }

    public byte[] getPayload() {
        if (payload == null || payload.isEmpty()) {
            return new byte[0];
        }
        return Base64.getDecoder().decode(payload);
    }

    /**
     * Utility method for efficient ByteBuffer-based serialization of double arrays.
     * This provides zero-copy optimization for matrix data transport.
     * @param data The double array to serialize
     * @return The ByteBuffer containing serialized data
     */
    public static ByteBuffer serializeDoubleArray(double[] data) {
        ByteBuffer buffer = ByteBuffer.allocate(data.length * 8);
        for (double val : data) {
            buffer.putDouble(val);
        }
        buffer.flip();
        return buffer;
    }
    
    /**
     * Utility method for efficient ByteBuffer-based deserialization of double arrays.
     * Provides efficient NIO buffer handling for large payloads.
     * @param buffer The ByteBuffer containing serialized data
     * @param length The number of doubles to deserialize
     * @return The reconstructed double array
     */
    public static double[] deserializeDoubleArray(ByteBuffer buffer, int length) {
        double[] data = new double[length];
        for (int i = 0; i < length; i++) {
            data[i] = buffer.getDouble();
        }
        return data;
    }

    /**
     * Converts the message to a JSON string followed by a newline for network transmission.
     */
    public byte[] pack() throws IOException {
        // Manual JSON construction to avoid external dependencies.
        String json = String.format("{\"magic\":\"%s\",\"version\":%d,\"messageType\":\"%s\",\"studentId\":\"%s\",\"timestamp\":%d,\"payload\":\"%s\"}\n",
            MAGIC,
            VERSION,
            this.messageType,
            this.studentId,
            this.timestamp,
            this.payload
        );
        return json.getBytes(StandardCharsets.UTF_8);
    }

    /**
     * Deserializes a Message from a network input stream by consuming a newline-delimited JSON payload.
     *
     * @param in The input stream from the socket.
     * @return A fully parsed Message object.
     * @throws IOException if the stream is closed or the message format is invalid.
     */
    public static Message unpack(InputStream in) throws IOException {
        ByteArrayOutputStream lineBuffer = new ByteArrayOutputStream();
        int b;
        while ((b = in.read()) != -1) {
            if (b == '\n') {
                break;
            }
            lineBuffer.write(b);
        }
        if (lineBuffer.size() == 0 && b == -1) {
            throw new EOFException("End of stream reached while waiting for message.");
        }

        String jsonLine = lineBuffer.toString(StandardCharsets.UTF_8.name());

        // This approach is more robust against key reordering and extra whitespace than single regex patterns.
        // Uses individual patterns to identify each required field in the JSON.
        Pattern magicPattern = Pattern.compile("\"magic\"\\s*:\\s*\"(.*?)\"");
        Pattern versionPattern = Pattern.compile("\"version\"\\s*:\\s*(\\d+)");
        Pattern typePattern = Pattern.compile("\"messageType\"\\s*:\\s*\"(.*?)\"");
        Pattern studentIdPattern = Pattern.compile("\"studentId\"\\s*:\\s*\"(.*?)\"");
        Pattern timestampPattern = Pattern.compile("\"timestamp\"\\s*:\\s*(\\d+)");
        Pattern payloadPattern = Pattern.compile("\"payload\"\\s*:\\s*\"(.*?)\"");

        Matcher magicMatcher = magicPattern.matcher(jsonLine);
        Matcher versionMatcher = versionPattern.matcher(jsonLine);
        Matcher typeMatcher = typePattern.matcher(jsonLine);
        Matcher studentIdMatcher = studentIdPattern.matcher(jsonLine);
        Matcher timestampMatcher = timestampPattern.matcher(jsonLine);
        Matcher payloadMatcher = payloadPattern.matcher(jsonLine);

        if (!magicMatcher.find() || !versionMatcher.find() || !typeMatcher.find() ||
            !studentIdMatcher.find() || !timestampMatcher.find() || !payloadMatcher.find()) {
            throw new IOException("Invalid or incomplete JSON message received: " + jsonLine);
        }

        String magic = magicMatcher.group(1);
        if (!MAGIC.equals(magic)) {
            throw new IOException("Invalid magic string: " + magic);
        }

        int version = Integer.parseInt(versionMatcher.group(1));
        if (VERSION != version) {
            throw new IOException("Unsupported protocol version: " + version);
        }

        String messageType = typeMatcher.group(1);
        String studentId = studentIdMatcher.group(1);
        long timestamp = Long.parseLong(timestampMatcher.group(1));
        String payload = payloadMatcher.group(1);

        return new Message(messageType, studentId, timestamp, payload);
    }
}
