#ifndef CTPPSDigi_TotemTimingEventInfo_h
#define CTPPSDigi_TotemTimingEventInfo_h

/** \class TotemTimingEventInfo
 *
 * Event Info Class for CTPPS Timing Detector
 *  
 * \author Mirko Berretti
 * \author Nicola Minafra
 * March 2018
 */

#include <cstdint>

class TotemTimingEventInfo{
 
  public:  
    TotemTimingEventInfo(const uint8_t hwId, const uint64_t L1ATimeStamp, const uint16_t bunchNumber, const uint32_t orbitNumber, const uint32_t eventNumber, const uint16_t channelMap, const uint16_t L1ALatency, const uint8_t numberOfSamples, const uint8_t offsetOfSamples );
    TotemTimingEventInfo(const TotemTimingEventInfo& eventInfo);
    TotemTimingEventInfo();
    ~TotemTimingEventInfo() {};
  
    /// Digis are equal if they have all the same values, NOT checking the samples!
    bool operator==(const TotemTimingEventInfo& eventInfo) const;

    /// Return digi values number
  
    /// Hardware Id formatted as: bits 0-3 Channel Id, bit 4 Sampic Id, bits 5-7 Digitizer Board Id
    inline unsigned int getHardwareId() const 
    { 
      return hwId_; 
    }
    
    inline unsigned int getHardwareBoardId() const
    {
      return (hwId_&0xE0)>>5;
    }
    
    inline unsigned int getHardwareSampicId() const
    {
      return (hwId_&0x10)>>4;
    }
    
    inline unsigned int getHardwareChannelId() const
    {
      return (hwId_&0x0F);
    }
    
    inline unsigned int getL1ATimeStamp() const
    {
      return L1ATimeStamp_;
    }
    
    inline unsigned int getBunchNumber() const
    {
      return bunchNumber_;
    }
    
    inline unsigned int getOrbitNumber() const
    {
      return orbitNumber_;
    }
    
    inline unsigned int getEventNumber() const
    {
      return eventNumber_;
    }
    
    inline uint16_t getChannelMap() const
    {
      return channelMap_;
    }
    
    inline unsigned int getL1ALatency() const
    {
      return L1ALatency_;
    }
    
    inline unsigned int getNumberOfSamples() const
    {
      return numberOfSamples_;
    }
    
    inline unsigned int getOffsetOfSamples() const
    {
      return offsetOfSamples_;
    }
    
       

    /// Set digi values
    /// Hardware Id formatted as: bits 0-3 Channel Id, bit 4 Sampic Id, bits 5-7 Digitizer Board Id
    inline void setHardwareId(const uint8_t hwId) 
    { 
      hwId_ = hwId; 
    }
    
    inline void setHardwareBoardId(const unsigned int BoardId)
    {
      hwId_ &= 0x1F;      // Set board bits to 0
      hwId_ |= ((BoardId&0x07)<<5) & 0xE0;
    }
    
    inline void setHardwareSampicId(const unsigned int SampicId)
    {
      hwId_ &= 0xEF;      // Set sampic bit to 0
      hwId_ |= ((SampicId&0x01)<<4) & 0x10;
    }
    
    inline void setHardwareChannelId(const unsigned int ChannelId)
    {
      hwId_ &= 0xF0;      // Set sampic bit to 0
      hwId_ |= (ChannelId&0x0F) & 0x0F;
    }
    
    inline void setL1ATimeStamp(const uint64_t L1ATimeStamp)
    {
      L1ATimeStamp_ = L1ATimeStamp;
    }
    
    inline void setBunchNumber(const uint16_t bunchNumber)
    {
      bunchNumber_ = bunchNumber;
    }
    
    inline void setOrbitNumber(const uint32_t orbitNumber)
    {
      orbitNumber_ = orbitNumber;
    }
    
    inline void setEventNumber(const uint32_t eventNumber)
    {
      eventNumber_ = eventNumber;
    }
    
    inline void setChannelMap(const uint16_t channelMap)
    {
      channelMap_ = channelMap;
    }
    
    inline void setL1ALatency(const uint16_t L1ALatency)
    {
      L1ALatency_ = L1ALatency;
    }
    
    inline void setNumberOfSamples(const uint8_t numberOfSamples)
    {
      numberOfSamples_ = numberOfSamples;
    }
    
    inline void setOffsetOfSamples(const uint8_t offsetOfSamples)
    {
      offsetOfSamples_ = offsetOfSamples;
    }
    
    
    
    


  private:
    uint8_t hwId_;
    uint64_t L1ATimeStamp_;
    uint16_t bunchNumber_;
    uint32_t orbitNumber_;
    uint32_t eventNumber_;
    uint16_t channelMap_;
    uint16_t L1ALatency_;
    uint8_t numberOfSamples_;
    uint8_t offsetOfSamples_;
    
};

#include <iostream>


inline bool operator< (const TotemTimingEventInfo& one, const TotemTimingEventInfo& other)
{
  if ( one.getEventNumber() < other.getEventNumber() )
    return true;
  if ( one.getL1ATimeStamp() < other.getL1ATimeStamp() )
    return true;
  if ( one.getHardwareId() < other.getHardwareId() )                                     
    return true; 
  return false;
}  


inline std::ostream & operator<<(std::ostream & o, const TotemTimingEventInfo& digi)
{
  return o << "TotemTimingEventInfo:"
	   << "\nHardwareId:\t" << std::hex << digi.getHardwareId()
           << "\nDB: " << std::dec << digi.getHardwareBoardId() << "\tSampic: " << digi.getHardwareSampicId() << "\tChannel: " << digi.getHardwareChannelId() 
           << "\nL1A Time Stamp:\t" << std::dec << digi.getL1ATimeStamp()
           << "\nL1A Latency:\t" << std::dec << digi.getL1ALatency()
           << "\nBunch Number:\t" << std::dec << digi.getBunchNumber()
           << "\nOrbit Number:\t" << std::dec << digi.getOrbitNumber()
           << "\nEvent Number:\t" << std::dec << digi.getEventNumber()
           << "\nChannels fired:\t" << std::hex << digi.getChannelMap()
           << "\nNumber of Samples:\t" << std::dec << digi.getNumberOfSamples()
           << "\nOffset of Samples:\t" << std::dec << digi.getOffsetOfSamples()
           << std::endl;
           
}

#endif

